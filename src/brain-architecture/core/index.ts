/**
 * 类人脑双系统全闭环AI架构 - 核心整合器
 * Human-Like Brain Dual-System Full-Loop AI Architecture - Core Orchestrator
 * 
 * 整合所有模块，提供统一的API接口
 */

import type {
  ArchitectureState,
  CycleExecutionState,
  MemoryAnchor,
  OptimizationMode,
  EvaluationReport,
  STDPUpdateRecord,
} from '../types';
import { REFRESH_CYCLE_MS, generateId } from '../config';
import { modelModificationManager } from '../modules/model';
import { inferenceEngine } from '../modules/engine';
import { stdpSystem } from '../modules/stdp';
import { closedLoopOptimization } from '../modules/optimizer';
import { hippocampusManager } from '../modules/hippocampus';
import { trainingManager } from '../modules/training';
import { evaluationSystem } from '../modules/evaluation';

// ==================== 架构状态管理 ====================

/**
 * 架构状态管理器
 */
export class ArchitectureStateManager {
  private state: ArchitectureState;
  private stateHistory: ArchitectureState[] = [];
  private maxHistoryLength: number = 100;

  constructor() {
    this.state = this.initializeState();
  }

  private initializeState(): ArchitectureState {
    return {
      isRunning: false,
      currentCycle: 0,
      currentPhase: 'complete',
      workingMemory: {
        currentContext: {
          currentToken: {
            tokenId: 0,
            embedding: new Float32Array(896),
            attentionWeights: new Float32Array(896),
            temporalFeature: new Float32Array(64),
            semanticFeature: new Float32Array(128),
            timestamp: Date.now(),
          },
          relatedMemoryAnchors: [],
          windowSize: 2,
        },
        recentOutputs: [],
        activeMemoryAnchors: [],
        cycleHistory: [],
        maxHistoryLength: 50,
      },
      stdpWeights: new Map(),
      hippocampus: {
        entorhinalCortex: {} as any,
        dentateGyrus: {} as any,
        ca3: {} as any,
        ca1: {} as any,
        sharpWaveRipple: {} as any,
      },
      optimizationMode: 'selfGeneration',
      deviceConfig: {
        device: 'raspberryPi4',
        quantization: {
          precision: 'INT4',
          maxMemoryMB: 420,
          maxComputeOverhead: 0.1,
        },
        maxMemoryMB: 420,
        targetLatencyMs: 10,
        offlineMode: true,
      },
    };
  }

  /**
   * 更新状态
   */
  updateState(updates: Partial<ArchitectureState>): void {
    this.state = {
      ...this.state,
      ...updates,
    };

    // 记录历史
    this.stateHistory.push({ ...this.state });
    if (this.stateHistory.length > this.maxHistoryLength) {
      this.stateHistory.shift();
    }
  }

  /**
   * 获取当前状态
   */
  getState(): ArchitectureState {
    return { ...this.state };
  }

  /**
   * 获取状态历史
   */
  getStateHistory(): ArchitectureState[] {
    return [...this.stateHistory];
  }
}

// ==================== 核心架构整合器 ====================

/**
 * 核心架构整合器
 * 统一协调所有模块的运行
 */
export class BrainArchitectureOrchestrator {
  private stateManager: ArchitectureStateManager;
  private isInitialized: boolean = false;
  private cycleTimer: ReturnType<typeof setInterval> | null = null;
  private eventListeners: Map<string, Set<(data: unknown) => void>> = new Map();

  constructor() {
    this.stateManager = new ArchitectureStateManager();
  }

  /**
   * 初始化架构
   */
  async initialize(): Promise<{
    success: boolean;
    message: string;
    memoryEstimate: {
      staticWeightsMB: number;
      dynamicWeightsMB: number;
      totalMB: number;
    };
  }> {
    console.log('[Orchestrator] 初始化类人脑双系统AI架构...');

    try {
      // 初始化训练模块
      const trainingReady = await trainingManager.initialize();
      if (!trainingReady) {
        return {
          success: false,
          message: '训练模块初始化失败',
          memoryEstimate: { staticWeightsMB: 0, dynamicWeightsMB: 0, totalMB: 0 },
        };
      }

      // 获取内存估算
      const memoryEstimate = modelModificationManager.getMemoryEstimate();

      this.isInitialized = true;

      this.stateManager.updateState({
        isRunning: false,
      });

      console.log('[Orchestrator] 架构初始化完成');

      return {
        success: true,
        message: '架构初始化成功',
        memoryEstimate,
      };
    } catch (error) {
      console.error('[Orchestrator] 初始化失败:', error);
      return {
        success: false,
        message: `初始化失败: ${error}`,
        memoryEstimate: { staticWeightsMB: 0, dynamicWeightsMB: 0, totalMB: 0 },
      };
    }
  }

  /**
   * 启动架构
   */
  start(): void {
    if (!this.isInitialized) {
      console.error('[Orchestrator] 请先初始化架构');
      return;
    }

    this.stateManager.updateState({ isRunning: true });
    inferenceEngine.start();

    // 启动海马体离线回放
    hippocampusManager.startOfflineReplay();

    console.log('[Orchestrator] 架构已启动');

    this.emit('started', { timestamp: Date.now() });
  }

  /**
   * 停止架构
   */
  stop(): void {
    this.stateManager.updateState({ isRunning: false });
    inferenceEngine.stop();

    // 停止海马体离线回放
    hippocampusManager.stopOfflineReplay();

    if (this.cycleTimer) {
      clearInterval(this.cycleTimer);
      this.cycleTimer = null;
    }

    console.log('[Orchestrator] 架构已停止');

    this.emit('stopped', { timestamp: Date.now() });
  }

  /**
   * 执行推理
   */
  async infer(input: string): Promise<{
    output: string;
    cycleStates: CycleExecutionState[];
    optimizationMode: OptimizationMode;
    memoryAnchors: MemoryAnchor[];
    stdpUpdates: STDPUpdateRecord[];
  }> {
    if (!this.stateManager.getState().isRunning) {
      this.start();
    }

    const cycleStates: CycleExecutionState[] = [];
    const allStdpUpdates: STDPUpdateRecord[] = [];

    // 执行自闭环优化
    const optimizationResult = closedLoopOptimization.execute(input);

    // 执行推理
    const output = await inferenceEngine.infer(
      input,
      [],
      (state) => {
        cycleStates.push(state);
        this.emit('cycleComplete', state);
      },
      (phase) => {
        this.stateManager.updateState({ currentPhase: phase });
        this.emit('phaseChange', phase);
      }
    );

    // 执行海马体记忆编码
    const memoryResult = hippocampusManager.executeCycle({
      tokenId: 0,
      embedding: new Float32Array(896).map(() => Math.random()),
      attentionWeights: new Float32Array(896),
      temporalFeature: new Float32Array(64),
      semanticFeature: new Float32Array(128),
      timestamp: Date.now(),
    });

    // 执行STDP更新
    const stdpUpdates = stdpSystem.executeUpdate(
      {
        tokenId: 0,
        embedding: new Float32Array(896).map(() => Math.random()),
        attentionWeights: new Float32Array(896),
        temporalFeature: new Float32Array(64),
        semanticFeature: new Float32Array(128),
        timestamp: Date.now(),
      },
      [],
      memoryResult.memoryAnchors,
      0.8
    );

    allStdpUpdates.push(...stdpUpdates, ...optimizationResult.stdpUpdates);

    // 更新状态
    this.stateManager.updateState({
      currentCycle: inferenceEngine.getCycleCount(),
      optimizationMode: optimizationResult.mode,
    });

    return {
      output: optimizationResult.result || output,
      cycleStates,
      optimizationMode: optimizationResult.mode,
      memoryAnchors: memoryResult.memoryAnchors,
      stdpUpdates: allStdpUpdates,
    };
  }

  /**
   * 执行测评
   */
  async evaluate(): Promise<EvaluationReport> {
    return evaluationSystem.runFullEvaluation();
  }

  /**
   * 获取架构状态
   */
  getState(): ArchitectureState {
    return this.stateManager.getState();
  }

  /**
   * 获取性能指标
   */
  getPerformanceMetrics() {
    return inferenceEngine.getPerformanceMetrics();
  }

  /**
   * 获取记忆统计
   */
  getMemoryStatistics() {
    return hippocampusManager.getMemoryStatistics();
  }

  /**
   * 获取STDP统计
   */
  getSTDPStatistics() {
    return stdpSystem.getWeightManager().getWeightStatistics();
  }

  /**
   * 获取优化模式
   */
  getCurrentOptimizationMode(): OptimizationMode {
    return closedLoopOptimization.getCurrentMode();
  }

  /**
   * 设置优化模式
   */
  setOptimizationMode(mode: OptimizationMode): void {
    closedLoopOptimization.setMode(mode);
    this.stateManager.updateState({ optimizationMode: mode });
  }

  /**
   * 执行场景训练
   */
  async trainScenario(scenarioName: string, epochs?: number) {
    return trainingManager.trainScenario(scenarioName, epochs);
  }

  /**
   * 获取可用训练场景
   */
  getAvailableScenarios(): string[] {
    return trainingManager.getScenarioTrainer().listScenarios();
  }

  /**
   * 注册事件监听
   */
  on(event: string, callback: (data: unknown) => void): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback);
  }

  /**
   * 移除事件监听
   */
  off(event: string, callback: (data: unknown) => void): void {
    this.eventListeners.get(event)?.delete(callback);
  }

  /**
   * 触发事件
   */
  private emit(event: string, data: any): void {
    this.eventListeners.get(event)?.forEach((callback) => {
      try {
        callback(data);
      } catch (error) {
        console.error(`[Orchestrator] 事件处理错误: ${event}`, error);
      }
    });
  }

  /**
   * 检查是否已初始化
   */
  isReady(): boolean {
    return this.isInitialized;
  }

  /**
   * 检查是否运行中
   */
  isRunning(): boolean {
    return this.stateManager.getState().isRunning;
  }
}

// 导出单例
export const brainArchitecture = new BrainArchitectureOrchestrator();

// 导出所有模块
export * from '../types';
export * from '../config';
export * from '../modules/model';
export * from '../modules/engine';
export * from '../modules/stdp';
export * from '../modules/optimizer';
export * from '../modules/hippocampus';
export * from '../modules/training';
export * from '../modules/evaluation';
