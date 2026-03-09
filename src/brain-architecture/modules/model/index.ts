/**
 * 模块1：Qwen3.5-0.8B底座模型基础改造
 * Module 1: Qwen3.5-0.8B Base Model Foundation Modification
 * 
 * 实现权重双轨拆分（90%静态 + 10%动态）
 * 实现原生接口适配改造
 */

import type {
  TokenFeature,
  MemoryAnchor,
  WeightBranch,
  AttentionLayerModified,
  FFNLayerModified,
  ModelModificationConfig,
} from '../../types';
import {
  WEIGHT_SPLIT_RATIO,
  QWEN_MODEL_CONFIG,
  generateId,
} from '../../config';

// ==================== 权重分支类 ====================

/**
 * 权重分支实现
 * 支持静态冻结分支和动态可更新分支
 */
export class WeightBranchImpl implements WeightBranch {
  type: 'static' | 'dynamic';
  ratio: number;
  weights: Map<string, Float32Array>;
  frozen: boolean;

  constructor(type: 'static' | 'dynamic', ratio: number, frozen: boolean) {
    this.type = type;
    this.ratio = ratio;
    this.weights = new Map();
    this.frozen = frozen;
  }

  /**
   * 初始化权重
   * 静态分支：继承预训练权重
   * 动态分支：小权重随机正态分布初始化
   */
  initializeWeights(
    layerId: string,
    shape: number[],
    pretrainedWeights?: Float32Array
  ): void {
    const size = shape.reduce((a, b) => a * b, 1);
    let weights: Float32Array;

    if (this.type === 'static' && pretrainedWeights) {
      // 静态分支：直接使用预训练权重
      weights = pretrainedWeights.slice(0, Math.floor(size * this.ratio));
    } else if (this.type === 'dynamic') {
      // 动态分支：小权重随机正态分布初始化
      weights = new Float32Array(Math.floor(size * this.ratio));
      for (let i = 0; i < weights.length; i++) {
        weights[i] = (Math.random() - 0.5) * 0.02; // 小权重初始化
      }
    } else {
      weights = new Float32Array(Math.floor(size * this.ratio));
    }

    this.weights.set(layerId, weights);
  }

  /**
   * 获取权重
   */
  getWeights(layerId: string): Float32Array | undefined {
    return this.weights.get(layerId);
  }

  /**
   * 更新权重（仅动态分支可更新）
   */
  updateWeights(layerId: string, delta: Float32Array): boolean {
    if (this.frozen) {
      console.warn(`[WeightBranch] 尝试更新冻结的静态权重: ${layerId}`);
      return false;
    }

    const currentWeights = this.weights.get(layerId);
    if (!currentWeights) {
      console.error(`[WeightBranch] 权重不存在: ${layerId}`);
      return false;
    }

    // 应用权重更新
    for (let i = 0; i < Math.min(currentWeights.length, delta.length); i++) {
      currentWeights[i] += delta[i];
    }

    return true;
  }

  /**
   * 冻结/解冻权重
   */
  setFrozen(frozen: boolean): void {
    if (this.type === 'static') {
      // 静态分支永远冻结
      this.frozen = true;
      console.warn('[WeightBranch] 静态分支权重永久冻结，无法解冻');
    } else {
      this.frozen = frozen;
    }
  }
}

// ==================== 注意力层改造 ====================

/**
 * 改造后的注意力层
 * 实现双轨权重 + 特征输出接口 + 海马体门控接口
 */
export class AttentionLayerModifiedImpl implements AttentionLayerModified {
  layerId: string;
  staticBranch: WeightBranch;
  dynamicBranch: WeightBranch;
  private lastFeatureOutput: TokenFeature | null = null;
  private memoryAnchors: MemoryAnchor[] = [];

  constructor(layerId: string, hiddenSize: number, numHeads: number) {
    this.layerId = layerId;

    // 创建静态分支（90%，冻结）
    this.staticBranch = new WeightBranchImpl(
      'static',
      WEIGHT_SPLIT_RATIO.static,
      true
    );

    // 创建动态分支（10%，可更新）
    this.dynamicBranch = new WeightBranchImpl(
      'dynamic',
      WEIGHT_SPLIT_RATIO.dynamic,
      false
    );

    // 初始化权重
    const headDim = hiddenSize / numHeads;
    this.staticBranch.initializeWeights(
      `${layerId}_qkv`,
      [hiddenSize, 3 * hiddenSize]
    );
    this.dynamicBranch.initializeWeights(
      `${layerId}_qkv_dynamic`,
      [hiddenSize, 3 * hiddenSize]
    );
    this.staticBranch.initializeWeights(
      `${layerId}_output`,
      [hiddenSize, hiddenSize]
    );
    this.dynamicBranch.initializeWeights(
      `${layerId}_output_dynamic`,
      [hiddenSize, hiddenSize]
    );
  }

  /**
   * 特征输出接口
   * 每个token推理完成后，输出注意力特征、时序特征、语义特征
   */
  featureOutputInterface(token: TokenFeature): TokenFeature {
    // 复用模型原生特征提取能力，无额外计算开销
    this.lastFeatureOutput = {
      ...token,
      timestamp: Date.now(),
    };

    return this.lastFeatureOutput;
  }

  /**
   * 海马体注意力门控接口
   * 在自注意力计算前，接入海马体输出的记忆锚点信号
   */
  hippocampusGateInterface(anchors: MemoryAnchor[]): void {
    this.memoryAnchors = anchors.slice(0, 2); // 最多2个记忆锚点
  }

  /**
   * 窄窗口注意力计算
   * 实现O(1)复杂度的注意力机制
   */
  computeNarrowWindowAttention(
    currentToken: TokenFeature,
    context: TokenFeature[]
  ): Float32Array {
    const hiddenSize = QWEN_MODEL_CONFIG.hiddenSize;
    const output = new Float32Array(hiddenSize);

    // 获取静态和动态权重
    const staticQKV = this.staticBranch.getWeights(`${this.layerId}_qkv`);
    const dynamicQKV = this.dynamicBranch.getWeights(`${this.layerId}_qkv_dynamic`);

    if (!staticQKV || !dynamicQKV) {
      return output;
    }

    // 窄窗口注意力：仅处理当前token + 1-2个上下文token
    const narrowContext = context.slice(-2); // 最近2个token

    // 计算注意力（简化实现）
    // 实际部署时会调用模型原生注意力计算
    let attentionSum = 0;
    for (let i = 0; i < hiddenSize; i++) {
      let val = currentToken.embedding[i] * staticQKV[i];
      
      // 加入动态权重贡献
      if (dynamicQKV[i]) {
        val += currentToken.embedding[i] * dynamicQKV[i] * 0.1;
      }

      // 加入记忆锚点引导
      for (const anchor of this.memoryAnchors) {
        if (anchor.featureVector[i]) {
          val += anchor.featureVector[i] * anchor.strength * 0.05;
        }
      }

      output[i] = val;
      attentionSum += Math.abs(val);
    }

    // 归一化
    if (attentionSum > 0) {
      for (let i = 0; i < hiddenSize; i++) {
        output[i] /= attentionSum;
      }
    }

    return output;
  }

  /**
   * 获取最后的特征输出
   */
  getLastFeatureOutput(): TokenFeature | null {
    return this.lastFeatureOutput;
  }

  /**
   * 获取当前记忆锚点
   */
  getMemoryAnchors(): MemoryAnchor[] {
    return this.memoryAnchors;
  }
}

// ==================== FFN层改造 ====================

/**
 * 改造后的FFN层
 * 实现双轨权重
 */
export class FFNLayerModifiedImpl implements FFNLayerModified {
  layerId: string;
  staticBranch: WeightBranch;
  dynamicBranch: WeightBranch;

  constructor(layerId: string, hiddenSize: number, intermediateSize: number) {
    this.layerId = layerId;

    // 创建静态分支（90%，冻结）
    this.staticBranch = new WeightBranchImpl(
      'static',
      WEIGHT_SPLIT_RATIO.static,
      true
    );

    // 创建动态分支（10%，可更新）
    this.dynamicBranch = new WeightBranchImpl(
      'dynamic',
      WEIGHT_SPLIT_RATIO.dynamic,
      false
    );

    // 初始化权重
    this.staticBranch.initializeWeights(
      `${layerId}_up`,
      [hiddenSize, intermediateSize]
    );
    this.dynamicBranch.initializeWeights(
      `${layerId}_up_dynamic`,
      [hiddenSize, intermediateSize]
    );
    this.staticBranch.initializeWeights(
      `${layerId}_down`,
      [intermediateSize, hiddenSize]
    );
    this.dynamicBranch.initializeWeights(
      `${layerId}_down_dynamic`,
      [intermediateSize, hiddenSize]
    );
    this.staticBranch.initializeWeights(
      `${layerId}_gate`,
      [hiddenSize, intermediateSize]
    );
    this.dynamicBranch.initializeWeights(
      `${layerId}_gate_dynamic`,
      [hiddenSize, intermediateSize]
    );
  }

  /**
   * FFN前向计算
   */
  forward(input: Float32Array): Float32Array {
    const intermediateSize = QWEN_MODEL_CONFIG.intermediateSize;
    const hiddenSize = QWEN_MODEL_CONFIG.hiddenSize;

    // 获取权重
    const staticUp = this.staticBranch.getWeights(`${this.layerId}_up`);
    const dynamicUp = this.dynamicBranch.getWeights(`${this.layerId}_up_dynamic`);
    const staticDown = this.staticBranch.getWeights(`${this.layerId}_down`);
    const dynamicDown = this.dynamicBranch.getWeights(`${this.layerId}_down_dynamic`);
    const staticGate = this.staticBranch.getWeights(`${this.layerId}_gate`);
    const dynamicGate = this.dynamicBranch.getWeights(`${this.layerId}_gate_dynamic`);

    // 简化的FFN计算（实际部署时使用模型原生计算）
    const intermediate = new Float32Array(intermediateSize);
    const output = new Float32Array(hiddenSize);

    // Up projection with gate
    for (let i = 0; i < intermediateSize; i++) {
      let val = 0;
      for (let j = 0; j < hiddenSize; j++) {
        const staticIdx = j * intermediateSize + i;
        if (staticUp && staticIdx < staticUp.length) {
          val += input[j] * staticUp[staticIdx];
        }
        if (dynamicUp && staticIdx < dynamicUp.length) {
          val += input[j] * dynamicUp[staticIdx] * 0.1;
        }
      }
      
      // Gate
      let gate = 1;
      for (let j = 0; j < hiddenSize; j++) {
        const gateIdx = j * intermediateSize + i;
        if (staticGate && gateIdx < staticGate.length) {
          gate *= Math.sigmoid(staticGate[gateIdx]);
        }
        if (dynamicGate && gateIdx < dynamicGate.length) {
          gate *= Math.sigmoid(dynamicGate[gateIdx] * 0.1);
        }
      }
      
      intermediate[i] = val * gate;
    }

    // Down projection
    for (let i = 0; i < hiddenSize; i++) {
      let val = 0;
      for (let j = 0; j < intermediateSize; j++) {
        const staticIdx = j * hiddenSize + i;
        if (staticDown && staticIdx < staticDown.length) {
          val += intermediate[j] * staticDown[staticIdx];
        }
        if (dynamicDown && staticIdx < dynamicDown.length) {
          val += intermediate[j] * dynamicDown[staticIdx] * 0.1;
        }
      }
      output[i] = val;
    }

    return output;
  }
}

// ==================== 模型改造管理器 ====================

/**
 * 模型改造管理器
 * 统一管理所有层的改造
 */
export class ModelModificationManager {
  private config: ModelModificationConfig;
  private attentionLayers: Map<string, AttentionLayerModifiedImpl> = new Map();
  private ffnLayers: Map<string, FFNLayerModifiedImpl> = new Map();
  private outputLayer: WeightBranchImpl;

  constructor() {
    this.config = {
      baseModel: 'Qwen3.5-0.8B',
      staticWeightRatio: 0.9,
      dynamicWeightRatio: 0.1,
      attentionLayers: [],
      ffnLayers: [],
      outputLayer: [],
    };

    // 初始化输出层
    this.outputLayer = new WeightBranchImpl('dynamic', 0.1, false);
    this.outputLayer.initializeWeights('output', [
      QWEN_MODEL_CONFIG.hiddenSize,
      QWEN_MODEL_CONFIG.vocabSize,
    ]);

    // 初始化所有层
    this.initializeLayers();
  }

  /**
   * 初始化所有层
   */
  private initializeLayers(): void {
    const { hiddenSize, numLayers, numAttentionHeads, intermediateSize } =
      QWEN_MODEL_CONFIG;

    // 初始化注意力层
    for (let i = 0; i < numLayers; i++) {
      const layerId = `attention_${i}`;
      const attentionLayer = new AttentionLayerModifiedImpl(
        layerId,
        hiddenSize,
        numAttentionHeads
      );
      this.attentionLayers.set(layerId, attentionLayer);
    }

    // 初始化FFN层
    for (let i = 0; i < numLayers; i++) {
      const layerId = `ffn_${i}`;
      const ffnLayer = new FFNLayerModifiedImpl(
        layerId,
        hiddenSize,
        intermediateSize
      );
      this.ffnLayers.set(layerId, ffnLayer);
    }

    console.log(
      `[ModelModification] 初始化完成: ${numLayers}个注意力层, ${numLayers}个FFN层`
    );
  }

  /**
   * 获取注意力层
   */
  getAttentionLayer(layerId: string): AttentionLayerModifiedImpl | undefined {
    return this.attentionLayers.get(layerId);
  }

  /**
   * 获取FFN层
   */
  getFFNLayer(layerId: string): FFNLayerModifiedImpl | undefined {
    return this.ffnLayers.get(layerId);
  }

  /**
   * 获取所有动态权重分支
   */
  getAllDynamicBranches(): WeightBranchImpl[] {
    const branches: WeightBranchImpl[] = [];

    this.attentionLayers.forEach((layer) => {
      branches.push(layer.dynamicBranch as WeightBranchImpl);
    });

    this.ffnLayers.forEach((layer) => {
      branches.push(layer.dynamicBranch as WeightBranchImpl);
    });

    branches.push(this.outputLayer);

    return branches;
  }

  /**
   * 获取模型配置
   */
  getConfig(): ModelModificationConfig {
    return this.config;
  }

  /**
   * 获取内存占用估算
   */
  getMemoryEstimate(): {
    staticWeightsMB: number;
    dynamicWeightsMB: number;
    totalMB: number;
  } {
    // Qwen3.5-0.8B 约800M参数
    // FP32: 800M * 4 bytes = 3200MB
    // INT4: 800M * 0.5 bytes = 400MB
    const totalParams = 800_000_000;
    const int4Size = totalParams * 0.5; // bytes

    const staticWeightsMB = (int4Size * 0.9) / (1024 * 1024);
    const dynamicWeightsMB = (int4Size * 0.1) / (1024 * 1024);
    const totalMB = staticWeightsMB + dynamicWeightsMB;

    return {
      staticWeightsMB: Math.round(staticWeightsMB),
      dynamicWeightsMB: Math.round(dynamicWeightsMB),
      totalMB: Math.round(totalMB),
    };
  }

  /**
   * 导出模型状态
   */
  exportState(): Record<string, unknown> {
    const state: Record<string, unknown> = {
      config: this.config,
      attentionLayers: {},
      ffnLayers: {},
      outputLayer: {},
    };

    this.attentionLayers.forEach((layer, id) => {
      state.attentionLayers[id] = {
        staticWeights: layer.staticBranch.weights,
        dynamicWeights: layer.dynamicBranch.weights,
      };
    });

    this.ffnLayers.forEach((layer, id) => {
      state.ffnLayers[id] = {
        staticWeights: layer.staticBranch.weights,
        dynamicWeights: layer.dynamicBranch.weights,
      };
    });

    return state;
  }
}

// 导出单例
export const modelModificationManager = new ModelModificationManager();
