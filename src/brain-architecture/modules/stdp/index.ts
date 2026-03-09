/**
 * 模块3：全链路STDP时序可塑性权重自动刷新系统
 * Module 3: Full-Chain STDP Temporal Plasticity Weight Auto-Refresh System
 * 
 * 实现Transformer原生适配的STDP机制
 * 全程无反向传播，纯本地时序信号驱动更新
 */

import type {
  STDPConfig,
  STDPWeight,
  STDPUpdateRecord,
  TokenFeature,
  MemoryAnchor,
} from '../../types';
import {
  DEFAULT_STDP_CONFIG,
  STDP_UPDATE_INTERVALS,
  generateId,
} from '../../config';

// ==================== STDP核心规则实现 ====================

/**
 * STDP规则计算器
 * 实现生物STDP机制的Transformer适配版本
 */
export class STDPRuleCalculator {
  private config: STDPConfig;
  private updateHistory: STDPUpdateRecord[] = [];
  private maxHistoryLength: number = 10000;

  constructor(config?: Partial<STDPConfig>) {
    this.config = {
      ...DEFAULT_STDP_CONFIG,
      ...config,
    };
  }

  /**
   * 计算LTP权重增强
   * 若前序token/上下文特征的激活时序早于当前token/神经元激活，
   * 且能有效支撑当前token的语义理解、逻辑推理、输出准确性，
   * 对应路径的STDP动态权重自动增强
   */
  calculateLTP(
    preActivationTime: number,
    postActivationTime: number,
    contributionScore: number,
    currentWeight: number
  ): number {
    // 时间差：前序激活早于后序激活
    const deltaTime = postActivationTime - preActivationTime;

    if (deltaTime <= 0) {
      // 时序不正确，不增强
      return 0;
    }

    // STDP时间窗口函数（指数衰减）
    const timeWindow = Math.exp(-deltaTime / 20); // 20ms时间窗口

    // LTP增强量 = 学习率 × 时间窗口 × 贡献度
    const deltaWeight =
      this.config.alpha * timeWindow * contributionScore * currentWeight;

    // 应用权重上限
    const newWeight = Math.min(
      currentWeight + deltaWeight,
      this.config.weightUpperBound
    );

    return newWeight - currentWeight;
  }

  /**
   * 计算LTD权重减弱
   * 若前序token/上下文特征的激活时序晚于当前token/神经元激活，
   * 或对当前输出无贡献、造成干扰与错误，
   * 对应路径的STDP动态权重自动减弱
   */
  calculateLTD(
    preActivationTime: number,
    postActivationTime: number,
    interferenceScore: number,
    currentWeight: number
  ): number {
    // 时间差：前序激活晚于后序激活（异常时序）
    const deltaTime = preActivationTime - postActivationTime;

    // LTD条件：时序错误 或 干扰分数高
    const shouldDepress = deltaTime > 0 || interferenceScore > 0.5;

    if (!shouldDepress) {
      return 0;
    }

    // STDP时间窗口函数（指数衰减）
    const effectiveDelta = Math.abs(deltaTime);
    const timeWindow = Math.exp(-effectiveDelta / 20);

    // LTD减弱量 = 学习率 × 时间窗口 × 干扰度
    const deltaWeight =
      -this.config.beta * timeWindow * interferenceScore * Math.abs(currentWeight);

    // 应用权重下限
    const newWeight = Math.max(
      currentWeight + deltaWeight,
      this.config.weightLowerBound
    );

    return newWeight - currentWeight;
  }

  /**
   * 计算综合STDP更新
   */
  calculateSTDPUpdate(
    preActivationTime: number,
    postActivationTime: number,
    contributionScore: number,
    interferenceScore: number,
    currentWeight: number
  ): { deltaWeight: number; updateType: 'LTP' | 'LTD' } {
    const ltpDelta = this.calculateLTP(
      preActivationTime,
      postActivationTime,
      contributionScore,
      currentWeight
    );

    const ltdDelta = this.calculateLTD(
      preActivationTime,
      postActivationTime,
      interferenceScore,
      currentWeight
    );

    const totalDelta = ltpDelta + ltdDelta;

    // 记录更新历史
    this.addUpdateHistory({
      timestamp: Date.now(),
      layerType: 'attention',
      weightId: generateId(),
      deltaValue: totalDelta,
      updateType: totalDelta > 0 ? 'LTP' : 'LTD',
      triggerReason: `贡献度:${contributionScore.toFixed(3)}, 干扰度:${interferenceScore.toFixed(3)}`,
    });

    return {
      deltaWeight: totalDelta,
      updateType: totalDelta > 0 ? 'LTP' : 'LTD',
    };
  }

  /**
   * 添加更新历史
   */
  private addUpdateHistory(record: STDPUpdateRecord): void {
    this.updateHistory.push(record);
    if (this.updateHistory.length > this.maxHistoryLength) {
      this.updateHistory.shift();
    }
  }

  /**
   * 获取更新历史
   */
  getUpdateHistory(): STDPUpdateRecord[] {
    return [...this.updateHistory];
  }

  /**
   * 获取配置
   */
  getConfig(): STDPConfig {
    return { ...this.config };
  }

  /**
   * 更新配置
   */
  updateConfig(config: Partial<STDPConfig>): void {
    this.config = {
      ...this.config,
      ...config,
    };
  }
}

// ==================== STDP权重管理器 ====================

/**
 * STDP权重管理器
 * 管理所有动态权重的存储和更新
 */
export class STDPWeightManager {
  private weights: Map<string, STDPWeight> = new Map();
  private calculator: STDPRuleCalculator;

  constructor(calculator: STDPRuleCalculator) {
    this.calculator = calculator;
  }

  /**
   * 注册权重
   */
  registerWeight(
    layerId: string,
    weightId: string,
    initialValue: Float32Array
  ): void {
    const key = `${layerId}_${weightId}`;
    this.weights.set(key, {
      layerId,
      weightId,
      value: initialValue,
      lastUpdateTime: Date.now(),
      updateCount: 0,
    });
  }

  /**
   * 获取权重
   */
  getWeight(layerId: string, weightId: string): STDPWeight | undefined {
    return this.weights.get(`${layerId}_${weightId}`);
  }

  /**
   * 更新权重
   */
  updateWeight(
    layerId: string,
    weightId: string,
    delta: Float32Array
  ): boolean {
    const weight = this.getWeight(layerId, weightId);
    if (!weight) {
      return false;
    }

    // 应用权重更新
    for (let i = 0; i < Math.min(weight.value.length, delta.length); i++) {
      weight.value[i] += delta[i];
    }

    weight.lastUpdateTime = Date.now();
    weight.updateCount++;

    return true;
  }

  /**
   * 获取所有权重
   */
  getAllWeights(): STDPWeight[] {
    return Array.from(this.weights.values());
  }

  /**
   * 获取权重统计
   */
  getWeightStatistics(): {
    totalWeights: number;
    totalUpdates: number;
    avgUpdateCount: number;
    lastUpdateTime: number;
  } {
    const allWeights = this.getAllWeights();
    const totalUpdates = allWeights.reduce((sum, w) => sum + w.updateCount, 0);

    return {
      totalWeights: allWeights.length,
      totalUpdates,
      avgUpdateCount: allWeights.length > 0 ? totalUpdates / allWeights.length : 0,
      lastUpdateTime: Math.max(...allWeights.map((w) => w.lastUpdateTime), 0),
    };
  }
}

// ==================== 全节点STDP更新系统 ====================

/**
 * 全节点STDP更新系统
 * 实现注意力层、FFN层、自评判、海马体门控的STDP更新
 */
export class FullNodeSTDPSystem {
  private calculator: STDPRuleCalculator;
  private weightManager: STDPWeightManager;
  private cycleCount: number = 0;
  private lastSelfEvaluationUpdate: number = 0;

  constructor(config?: Partial<STDPConfig>) {
    this.calculator = new STDPRuleCalculator(config);
    this.weightManager = new STDPWeightManager(this.calculator);
    this.initializeWeights();
  }

  /**
   * 初始化所有动态权重
   */
  private initializeWeights(): void {
    // 初始化注意力层动态权重
    for (let i = 0; i < 24; i++) {
      // Qwen3.5-0.8B有24层
      const layerId = `attention_${i}`;
      this.weightManager.registerWeight(
        layerId,
        'qkv_dynamic',
        new Float32Array(896 * 3 * 0.1).map(() => (Math.random() - 0.5) * 0.02)
      );
      this.weightManager.registerWeight(
        layerId,
        'output_dynamic',
        new Float32Array(896 * 0.1).map(() => (Math.random() - 0.5) * 0.02)
      );
    }

    // 初始化FFN层动态权重
    for (let i = 0; i < 24; i++) {
      const layerId = `ffn_${i}`;
      this.weightManager.registerWeight(
        layerId,
        'up_dynamic',
        new Float32Array(896 * 4864 * 0.1).map(() => (Math.random() - 0.5) * 0.02)
      );
      this.weightManager.registerWeight(
        layerId,
        'down_dynamic',
        new Float32Array(4864 * 896 * 0.1).map(() => (Math.random() - 0.5) * 0.02)
      );
    }

    // 初始化海马体门控权重
    this.weightManager.registerWeight(
      'hippocampus',
      'gate_dynamic',
      new Float32Array(64 * 896).map(() => (Math.random() - 0.5) * 0.02)
    );

    console.log('[STDP] 动态权重初始化完成');
  }

  /**
   * 执行全节点STDP更新
   * 每个刷新周期调用
   */
  executeUpdate(
    currentToken: TokenFeature,
    contextTokens: TokenFeature[],
    memoryAnchors: MemoryAnchor[],
    outputQuality: number
  ): STDPUpdateRecord[] {
    this.cycleCount++;
    const updates: STDPUpdateRecord[] = [];
    const now = Date.now();

    // 1. 注意力层STDP更新（每个周期）
    const attentionUpdates = this.updateAttentionLayer(
      currentToken,
      contextTokens,
      now
    );
    updates.push(...attentionUpdates);

    // 2. FFN层STDP更新（每个周期）
    const ffnUpdates = this.updateFFNLayer(currentToken, now);
    updates.push(...ffnUpdates);

    // 3. 自评判STDP更新（每10个周期）
    if (
      this.cycleCount - this.lastSelfEvaluationUpdate >=
      STDP_UPDATE_INTERVALS.selfEvaluation
    ) {
      const selfEvalUpdates = this.updateSelfEvaluation(outputQuality, now);
      updates.push(...selfEvalUpdates);
      this.lastSelfEvaluationUpdate = this.cycleCount;
    }

    // 4. 海马体门控STDP更新（每个周期）
    const hippocampusUpdates = this.updateHippocampusGate(
      currentToken,
      memoryAnchors,
      now
    );
    updates.push(...hippocampusUpdates);

    return updates;
  }

  /**
   * 注意力层STDP更新
   * 根据窄窗口内上下文与当前token的时序关联、语义贡献度更新
   */
  private updateAttentionLayer(
    currentToken: TokenFeature,
    contextTokens: TokenFeature[],
    now: number
  ): STDPUpdateRecord[] {
    const updates: STDPUpdateRecord[] = [];

    for (const contextToken of contextTokens) {
      // 计算时序关联
      const timeDiff = currentToken.timestamp - contextToken.timestamp;

      // 计算语义贡献度（基于特征相似度）
      const similarity = this.calculateSimilarity(
        currentToken.embedding,
        contextToken.embedding
      );

      // 计算STDP更新
      const { deltaWeight, updateType } = this.calculator.calculateSTDPUpdate(
        contextToken.timestamp,
        currentToken.timestamp,
        similarity, // 贡献度
        1 - similarity, // 干扰度
        0.5 // 当前权重
      );

      updates.push({
        timestamp: now,
        layerType: 'attention',
        weightId: `attention_context_${contextToken.tokenId}`,
        deltaValue: deltaWeight,
        updateType,
        triggerReason: `时序关联:${timeDiff}ms, 语义贡献:${similarity.toFixed(3)}`,
      });
    }

    return updates;
  }

  /**
   * FFN层STDP更新
   * 对高频特征、专属术语、用户习惯表达自动增强
   */
  private updateFFNLayer(
    currentToken: TokenFeature,
    now: number
  ): STDPUpdateRecord[] {
    const updates: STDPUpdateRecord[] = [];

    // 分析当前token的特征强度
    const featureStrength = this.calculateFeatureStrength(currentToken.embedding);

    // 根据特征强度决定更新方向
    const { deltaWeight, updateType } = this.calculator.calculateSTDPUpdate(
      now - 10, // 前序时间
      now, // 当前时间
      featureStrength, // 贡献度
      0, // 干扰度
      0.5
    );

    updates.push({
      timestamp: now,
      layerType: 'ffn',
      weightId: 'ffn_feature_dynamic',
      deltaValue: deltaWeight,
      updateType,
      triggerReason: `特征强度:${featureStrength.toFixed(3)}`,
    });

    return updates;
  }

  /**
   * 自评判STDP更新
   * 根据评判结果更新正确/错误路径的权重
   */
  private updateSelfEvaluation(
    outputQuality: number,
    now: number
  ): STDPUpdateRecord[] {
    const updates: STDPUpdateRecord[] = [];

    // 根据输出质量决定更新方向
    const isGood = outputQuality > 0.7;

    const { deltaWeight, updateType } = this.calculator.calculateSTDPUpdate(
      now - 100, // 前序时间
      now,
      isGood ? outputQuality : 0, // 好结果增强
      isGood ? 0 : 1 - outputQuality, // 坏结果减弱
      0.5
    );

    updates.push({
      timestamp: now,
      layerType: 'selfEvaluation',
      weightId: 'self_eval_path',
      deltaValue: deltaWeight,
      updateType,
      triggerReason: `输出质量:${outputQuality.toFixed(3)}`,
    });

    return updates;
  }

  /**
   * 海马体门控STDP更新
   * 对推理有正向贡献的记忆锚点增强，无效锚点减弱
   */
  private updateHippocampusGate(
    currentToken: TokenFeature,
    memoryAnchors: MemoryAnchor[],
    now: number
  ): STDPUpdateRecord[] {
    const updates: STDPUpdateRecord[] = [];

    for (const anchor of memoryAnchors) {
      // 计算记忆锚点与当前token的相关性
      const relevance = this.calculateSimilarity(
        currentToken.embedding,
        anchor.featureVector
      );

      // 计算记忆锚点的贡献度
      const contribution = relevance * anchor.strength;

      const { deltaWeight, updateType } = this.calculator.calculateSTDPUpdate(
        anchor.timestamp,
        now,
        contribution,
        1 - relevance,
        anchor.strength
      );

      updates.push({
        timestamp: now,
        layerType: 'hippocampusGate',
        weightId: anchor.id,
        deltaValue: deltaWeight,
        updateType,
        triggerReason: `记忆相关性:${relevance.toFixed(3)}, 贡献度:${contribution.toFixed(3)}`,
      });
    }

    return updates;
  }

  /**
   * 计算向量相似度（余弦相似度）
   */
  private calculateSimilarity(a: Float32Array, b: Float32Array): number {
    const minLength = Math.min(a.length, b.length);
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < minLength; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * 计算特征强度
   */
  private calculateFeatureStrength(embedding: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < embedding.length; i++) {
      sum += Math.abs(embedding[i]);
    }
    return sum / embedding.length;
  }

  /**
   * 获取计算器
   */
  getCalculator(): STDPRuleCalculator {
    return this.calculator;
  }

  /**
   * 获取权重管理器
   */
  getWeightManager(): STDPWeightManager {
    return this.weightManager;
  }

  /**
   * 获取周期计数
   */
  getCycleCount(): number {
    return this.cycleCount;
  }

  /**
   * 获取更新历史
   */
  getUpdateHistory(): STDPUpdateRecord[] {
    return this.calculator.getUpdateHistory();
  }
}

// 导出单例
export const stdpSystem = new FullNodeSTDPSystem();
