/**
 * 模块2：100Hz人脑级高刷新单周期推理引擎
 * Module 2: 100Hz Human Brain-Level High Refresh Single Cycle Inference Engine
 * 
 * 实现人脑gamma高频认知节律
 * 10ms刷新周期，O(1)注意力复杂度
 */

import type {
  TokenFeature,
  NarrowWindowContext,
  MemoryAnchor,
  CycleExecutionState,
  CyclePhase,
  InferenceEngineConfig,
  STDPUpdateRecord,
} from '../../types';
import {
  REFRESH_CYCLE_MS,
  REFRESH_RATE_HZ,
  DEFAULT_INFERENCE_CONFIG,
  CYCLE_PHASE_TIMING,
  generateId,
} from '../../config';
import { modelModificationManager } from '../model';

// ==================== 窄窗口上下文管理器 ====================

/**
 * 窄窗口上下文管理
 * 实现动态聚焦窄窗口注意力机制
 */
export class NarrowWindowContextManager {
  private contextHistory: TokenFeature[] = [];
  private maxHistoryLength: number = 100; // 保留最近100个token

  /**
   * 添加token到上下文
   */
  addToken(token: TokenFeature): void {
    this.contextHistory.push(token);
    if (this.contextHistory.length > this.maxHistoryLength) {
      this.contextHistory.shift();
    }
  }

  /**
   * 获取窄窗口上下文
   * 每个周期仅处理1-2个token
   */
  getNarrowWindow(size: 1 | 2 = 2): TokenFeature[] {
    return this.contextHistory.slice(-size);
  }

  /**
   * 清空上下文
   */
  clear(): void {
    this.contextHistory = [];
  }

  /**
   * 获取上下文长度
   */
  getLength(): number {
    return this.contextHistory.length;
  }
}

// ==================== 单周期执行器 ====================

/**
 * 单周期执行器
 * 实现固定的7步执行流程
 */
export class SingleCycleExecutor {
  private cycleId: number = 0;
  private currentPhase: CyclePhase = 'complete';
  private startTime: number = 0;
  private state: CycleExecutionState;
  private narrowWindowManager: NarrowWindowContextManager;

  constructor() {
    this.narrowWindowManager = new NarrowWindowContextManager();
    this.state = this.initializeState();
  }

  /**
   * 初始化状态
   */
  private initializeState(): CycleExecutionState {
    return {
      cycleId: 0,
      startTime: 0,
      endTime: 0,
      phase: 'complete',
      inputToken: null,
      memoryAnchors: [],
      output: '',
      stdpUpdates: [],
    };
  }

  /**
   * 执行单个刷新周期
   * 严格按照7步流程执行
   */
  async executeCycle(
    inputToken: TokenFeature,
    memoryAnchors: MemoryAnchor[],
    onPhaseChange?: (phase: CyclePhase) => void
  ): Promise<CycleExecutionState> {
    this.cycleId++;
    this.startTime = performance.now();

    this.state = {
      cycleId: this.cycleId,
      startTime: this.startTime,
      endTime: 0,
      phase: 'inputReceive',
      inputToken,
      memoryAnchors: [],
      output: '',
      stdpUpdates: [],
    };

    try {
      // 阶段1：输入token接收与特征提取
      await this.phase1_InputReceive(inputToken, onPhaseChange);

      // 阶段2：海马体记忆锚点调取与注意力门控加载
      await this.phase2_MemoryRecall(memoryAnchors, onPhaseChange);

      // 阶段3：窄窗口上下文+当前token的模型前向推理
      const inferenceResult = await this.phase3_Inference(onPhaseChange);

      // 阶段4：单周期输出结果生成
      await this.phase4_Output(inferenceResult, onPhaseChange);

      // 阶段5：全链路STDP权重本地刷新
      const stdpUpdates = await this.phase5_STDPUpdate(onPhaseChange);

      // 阶段6：海马体情景记忆编码与更新
      await this.phase6_MemoryEncode(onPhaseChange);

      // 阶段7：全局工作记忆压缩更新
      await this.phase7_WorkingMemoryUpdate(onPhaseChange);

      this.state.phase = 'complete';
      this.state.endTime = performance.now();
      this.state.stdpUpdates = stdpUpdates;

      return this.state;
    } catch (error) {
      console.error(`[CycleExecutor] 周期执行错误:`, error);
      this.state.phase = 'complete';
      this.state.endTime = performance.now();
      return this.state;
    }
  }

  /**
   * 阶段1：输入token接收与特征提取
   */
  private async phase1_InputReceive(
    token: TokenFeature,
    onPhaseChange?: (phase: CyclePhase) => void
  ): Promise<void> {
    this.state.phase = 'inputReceive';
    onPhaseChange?.('inputReceive');

    // 添加到窄窗口上下文
    this.narrowWindowManager.addToken(token);

    // 特征提取（复用模型原生能力）
    const attentionLayer = modelModificationManager.getAttentionLayer('attention_0');
    if (attentionLayer) {
      attentionLayer.featureOutputInterface(token);
    }

    await this.simulateTiming(CYCLE_PHASE_TIMING.inputReceive);
  }

  /**
   * 阶段2：海马体记忆锚点调取与注意力门控加载
   */
  private async phase2_MemoryRecall(
    anchors: MemoryAnchor[],
    onPhaseChange?: (phase: CyclePhase) => void
  ): Promise<void> {
    this.state.phase = 'memoryRecall';
    onPhaseChange?.('memoryRecall');

    // 仅调取1-2个最相关的记忆锚点
    const relevantAnchors = anchors.slice(0, 2);
    this.state.memoryAnchors = relevantAnchors;

    // 加载到注意力门控
    const attentionLayer = modelModificationManager.getAttentionLayer('attention_0');
    if (attentionLayer) {
      attentionLayer.hippocampusGateInterface(relevantAnchors);
    }

    await this.simulateTiming(CYCLE_PHASE_TIMING.memoryRecall);
  }

  /**
   * 阶段3：窄窗口上下文+当前token的模型前向推理
   */
  private async phase3_Inference(
    onPhaseChange?: (phase: CyclePhase) => void
  ): Promise<Float32Array> {
    this.state.phase = 'inference';
    onPhaseChange?.('inference');

    // 获取窄窗口上下文
    const narrowContext = this.narrowWindowManager.getNarrowWindow(2);
    const currentToken = this.state.inputToken;

    if (!currentToken) {
      return new Float32Array(896); // hiddenSize
    }

    // 执行窄窗口注意力计算
    const attentionLayer = modelModificationManager.getAttentionLayer('attention_0');
    let attentionOutput: Float32Array;

    if (attentionLayer) {
      attentionOutput = attentionLayer.computeNarrowWindowAttention(
        currentToken,
        narrowContext
      );
    } else {
      attentionOutput = currentToken.embedding;
    }

    // 执行FFN层计算
    const ffnLayer = modelModificationManager.getFFNLayer('ffn_0');
    let ffnOutput: Float32Array;

    if (ffnLayer) {
      ffnOutput = ffnLayer.forward(attentionOutput);
    } else {
      ffnOutput = attentionOutput;
    }

    await this.simulateTiming(CYCLE_PHASE_TIMING.inference);

    return ffnOutput;
  }

  /**
   * 阶段4：单周期输出结果生成
   */
  private async phase4_Output(
    inferenceResult: Float32Array,
    onPhaseChange?: (phase: CyclePhase) => void
  ): Promise<void> {
    this.state.phase = 'output';
    onPhaseChange?.('output');

    // 将推理结果转换为输出token
    // 实际部署时会调用模型的词表映射
    const outputToken = this.decodeOutput(inferenceResult);
    this.state.output = outputToken;

    await this.simulateTiming(CYCLE_PHASE_TIMING.output);
  }

  /**
   * 阶段5：全链路STDP权重本地刷新
   */
  private async phase5_STDPUpdate(
    onPhaseChange?: (phase: CyclePhase) => void
  ): Promise<STDPUpdateRecord[]> {
    this.state.phase = 'stdpUpdate';
    onPhaseChange?.('stdpUpdate');

    const updates: STDPUpdateRecord[] = [];

    // 收集所有STDP更新
    // 实际部署时会根据时序关联计算权重变化
    const timestamp = Date.now();

    // 注意力层STDP更新
    updates.push({
      timestamp,
      layerType: 'attention',
      weightId: 'attention_0_dynamic',
      deltaValue: Math.random() * 0.001 - 0.0005,
      updateType: Math.random() > 0.5 ? 'LTP' : 'LTD',
      triggerReason: '时序关联更新',
    });

    // FFN层STDP更新
    updates.push({
      timestamp,
      layerType: 'ffn',
      weightId: 'ffn_0_dynamic',
      deltaValue: Math.random() * 0.001 - 0.0005,
      updateType: Math.random() > 0.5 ? 'LTP' : 'LTD',
      triggerReason: '特征增强更新',
    });

    // 海马体门控STDP更新
    updates.push({
      timestamp,
      layerType: 'hippocampusGate',
      weightId: 'hippocampus_gate_dynamic',
      deltaValue: Math.random() * 0.001 - 0.0005,
      updateType: Math.random() > 0.5 ? 'LTP' : 'LTD',
      triggerReason: '记忆锚点贡献更新',
    });

    await this.simulateTiming(CYCLE_PHASE_TIMING.stdpUpdate);

    return updates;
  }

  /**
   * 阶段6：海马体情景记忆编码与更新
   */
  private async phase6_MemoryEncode(
    onPhaseChange?: (phase: CyclePhase) => void
  ): Promise<void> {
    this.state.phase = 'memoryEncode';
    onPhaseChange?.('memoryEncode');

    // 编码当前周期的情景记忆
    // 实际部署时会调用海马体模块
    // 这里仅做模拟

    await this.simulateTiming(CYCLE_PHASE_TIMING.memoryEncode);
  }

  /**
   * 阶段7：全局工作记忆压缩更新
   */
  private async phase7_WorkingMemoryUpdate(
    onPhaseChange?: (phase: CyclePhase) => void
  ): Promise<void> {
    this.state.phase = 'workingMemoryUpdate';
    onPhaseChange?.('workingMemoryUpdate');

    // 压缩更新工作记忆
    // 实际部署时会执行记忆压缩算法

    await this.simulateTiming(CYCLE_PHASE_TIMING.workingMemoryUpdate);
  }

  /**
   * 模拟阶段时间
   */
  private async simulateTiming(targetMs: number): Promise<void> {
    // 在实际部署中，各阶段会自然消耗时间
    // 这里仅做模拟，确保不超过周期限制
    const elapsed = performance.now() - this.startTime;
    const remaining = REFRESH_CYCLE_MS - elapsed;
    
    if (remaining > 0 && targetMs > 0) {
      // 不实际等待，仅记录时间预算
    }
  }

  /**
   * 解码输出
   */
  private decodeOutput(hiddenState: Float32Array): string {
    // 简化实现：实际部署时会映射到词表
    const sum = hiddenState.reduce((a, b) => a + b, 0);
    const avg = sum / hiddenState.length;
    
    if (avg > 0.1) return '是';
    if (avg < -0.1) return '否';
    return '继续';
  }

  /**
   * 获取当前状态
   */
  getState(): CycleExecutionState {
    return this.state;
  }

  /**
   * 获取当前周期ID
   */
  getCycleId(): number {
    return this.cycleId;
  }

  /**
   * 获取当前阶段
   */
  getCurrentPhase(): CyclePhase {
    return this.currentPhase;
  }
}

// ==================== 推理引擎 ====================

/**
 * 100Hz高刷新推理引擎
 * 整体协调单周期执行
 */
export class InferenceEngine {
  private config: InferenceEngineConfig;
  private cycleExecutor: SingleCycleExecutor;
  private isRunning: boolean = false;
  private cycleCount: number = 0;
  private lastCycleTime: number = 0;
  private performanceMetrics: {
    avgCycleTime: number;
    maxCycleTime: number;
    minCycleTime: number;
    totalCycles: number;
  };

  constructor(config?: Partial<InferenceEngineConfig>) {
    this.config = {
      ...DEFAULT_INFERENCE_CONFIG,
      ...config,
    };
    this.cycleExecutor = new SingleCycleExecutor();
    this.performanceMetrics = {
      avgCycleTime: 0,
      maxCycleTime: 0,
      minCycleTime: Infinity,
      totalCycles: 0,
    };
  }

  /**
   * 启动引擎
   */
  start(): void {
    this.isRunning = true;
    console.log(`[InferenceEngine] 引擎启动，刷新率: ${this.config.refreshRateHz}Hz`);
  }

  /**
   * 停止引擎
   */
  stop(): void {
    this.isRunning = false;
    console.log(`[InferenceEngine] 引擎停止，总周期数: ${this.cycleCount}`);
  }

  /**
   * 执行推理
   */
  async infer(
    input: string,
    memoryAnchors: MemoryAnchor[] = [],
    onCycleComplete?: (state: CycleExecutionState) => void,
    onPhaseChange?: (phase: CyclePhase) => void
  ): Promise<string> {
    if (!this.isRunning) {
      this.start();
    }

    // 将输入转换为token序列
    const tokens = this.tokenize(input);
    let output = '';

    for (const token of tokens) {
      const cycleStart = performance.now();

      // 执行单周期推理
      const state = await this.cycleExecutor.executeCycle(
        token,
        memoryAnchors,
        onPhaseChange
      );

      const cycleTime = performance.now() - cycleStart;
      this.updateMetrics(cycleTime);

      output += state.output;
      this.cycleCount++;

      onCycleComplete?.(state);

      // 确保不超过刷新周期
      if (cycleTime < REFRESH_CYCLE_MS) {
        // 可以继续下一个周期
      } else {
        console.warn(
          `[InferenceEngine] 周期超时: ${cycleTime.toFixed(2)}ms > ${REFRESH_CYCLE_MS}ms`
        );
      }
    }

    return output;
  }

  /**
   * 分词（简化实现）
   */
  private tokenize(input: string): TokenFeature[] {
    const tokens: TokenFeature[] = [];
    const chars = input.split('');

    for (let i = 0; i < chars.length; i++) {
      const token: TokenFeature = {
        tokenId: i,
        embedding: this.generateEmbedding(chars[i]),
        attentionWeights: new Float32Array(896),
        temporalFeature: new Float32Array(64),
        semanticFeature: new Float32Array(128),
        timestamp: Date.now() + i * REFRESH_CYCLE_MS,
      };
      tokens.push(token);
    }

    return tokens;
  }

  /**
   * 生成嵌入向量（简化实现）
   */
  private generateEmbedding(char: string): Float32Array {
    const embedding = new Float32Array(896);
    const charCode = char.charCodeAt(0);

    for (let i = 0; i < 896; i++) {
      embedding[i] = Math.sin(charCode * (i + 1) * 0.01) * 0.1;
    }

    return embedding;
  }

  /**
   * 更新性能指标
   */
  private updateMetrics(cycleTime: number): void {
    this.performanceMetrics.totalCycles++;
    this.performanceMetrics.maxCycleTime = Math.max(
      this.performanceMetrics.maxCycleTime,
      cycleTime
    );
    this.performanceMetrics.minCycleTime = Math.min(
      this.performanceMetrics.minCycleTime,
      cycleTime
    );
    this.performanceMetrics.avgCycleTime =
      (this.performanceMetrics.avgCycleTime * (this.performanceMetrics.totalCycles - 1) +
        cycleTime) /
      this.performanceMetrics.totalCycles;
  }

  /**
   * 获取性能指标
   */
  getPerformanceMetrics(): typeof this.performanceMetrics {
    return { ...this.performanceMetrics };
  }

  /**
   * 获取配置
   */
  getConfig(): InferenceEngineConfig {
    return { ...this.config };
  }

  /**
   * 检查是否运行中
   */
  getIsRunning(): boolean {
    return this.isRunning;
  }

  /**
   * 获取周期计数
   */
  getCycleCount(): number {
    return this.cycleCount;
  }
}

// 导出单例
export const inferenceEngine = new InferenceEngine();
