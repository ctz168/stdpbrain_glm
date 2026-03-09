/**
 * 类人脑双系统全闭环AI架构 - 核心类型定义
 * Human-Like Brain Dual-System Full-Loop AI Architecture
 * 
 * 基于 Qwen3.5-0.8B 底座模型
 * 实现100Hz人脑级高刷新推理 + STDP学习机制
 */

// ==================== 基础类型 ====================

/** 刷新周期：10ms = 100Hz */
export type RefreshCycle = 10; // ms

/** 权重拆分比例 */
export interface WeightSplitRatio {
  static: 0.9;   // 90%静态基础权重
  dynamic: 0.1;  // 10%STDP动态增量权重
}

/** 量化配置 */
export interface QuantizationConfig {
  precision: 'INT4';
  maxMemoryMB: 420;  // 最大显存占用
  maxComputeOverhead: 0.1;  // 最大算力开销比例
}

// ==================== Token与特征类型 ====================

/** Token特征向量 */
export interface TokenFeature {
  tokenId: number;
  embedding: Float32Array;  // 嵌入向量
  attentionWeights: Float32Array;  // 注意力权重
  temporalFeature: Float32Array;  // 时序特征
  semanticFeature: Float32Array;  // 语义特征
  timestamp: number;  // 时间戳(ms)
}

/** 窄窗口上下文 */
export interface NarrowWindowContext {
  currentToken: TokenFeature;
  relatedMemoryAnchors: MemoryAnchor[];  // 1-2个相关记忆锚点
  windowSize: 1 | 2;  // 窄窗口大小
}

// ==================== STDP权重类型 ====================

/** STDP权重配置 */
export interface STDPConfig {
  alpha: number;  // LTP学习率 (权重增强)
  beta: number;   // LTD学习率 (权重减弱)
  updateThreshold: number;  // 更新阈值
  weightUpperBound: number;  // 权重上限
  weightLowerBound: number;  // 权重下限
}

/** STDP动态权重 */
export interface STDPWeight {
  layerId: string;
  weightId: string;
  value: Float32Array;
  lastUpdateTime: number;
  updateCount: number;
}

/** STDP更新记录 */
export interface STDPUpdateRecord {
  timestamp: number;
  layerType: 'attention' | 'ffn' | 'selfEvaluation' | 'hippocampusGate';
  weightId: string;
  deltaValue: number;  // 权重变化量
  updateType: 'LTP' | 'LTD';  // 增强或减弱
  triggerReason: string;
}

// ==================== 海马体记忆类型 ====================

/** 记忆锚点 */
export interface MemoryAnchor {
  id: string;
  timestamp: number;  // 10ms级时间戳
  featureVector: Float32Array;  // 64维特征向量
  temporalSkeleton: number[];  // 时序骨架
  semanticPointer: string;  // 语义指针
  causalRelation: string;  // 因果关联
  accessCount: number;  // 访问次数
  lastAccessTime: number;
  strength: number;  // 记忆强度
}

/** 海马体分区模块 */
export interface HippocampusModule {
  // 内嗅皮层EC - 特征编码单元
  entorhinalCortex: EntorhinalCortexUnit;
  // 齿状回DG - 模式分离单元
  dentateGyrus: DentateGyrusUnit;
  // CA3区 - 情景记忆库+模式补全
  ca3: CA3Unit;
  // CA1区 - 时序编码+注意力门控
  ca1: CA1Unit;
  // 尖波涟漪SWR - 离线回放巩固
  sharpWaveRipple: SWRUnit;
}

/** 内嗅皮层EC单元 */
export interface EntorhinalCortexUnit {
  encodeFeature(tokenFeature: TokenFeature): Float32Array;
  normalizeSparse(feature: Float32Array): Float32Array;
}

/** 齿状回DG单元 */
export interface DentateGyrusUnit {
  patternSeparation(feature: Float32Array): string;  // 返回唯一记忆ID
  orthogonalProjection(feature: Float32Array): Float32Array;
}

/** CA3区单元 */
export interface CA3Unit {
  episodicMemoryStore: EpisodicMemoryStore;
  patternCompletion(partialCue: Float32Array): MemoryAnchor[];
  storeMemory(memory: MemoryAnchor): void;
}

/** 情景记忆存储 */
export interface EpisodicMemoryStore {
  memories: Map<string, MemoryAnchor>;
  maxSize: number;  // 最大存储数量
  currentSize: number;
}

/** CA1区单元 */
export interface CA1Unit {
  temporalEncoding(memory: MemoryAnchor): void;
  attentionGateControl(currentContext: NarrowWindowContext): MemoryAnchor[];
  outputMemoryAnchors(count: 1 | 2): MemoryAnchor[];
}

/** 尖波涟漪SWR单元 */
export interface SWRUnit {
  replayConsolidation(): void;
  memoryPruning(): void;
  weightOptimization(): void;
}

// ==================== 自闭环优化类型 ====================

/** 运行模式 */
export type OptimizationMode = 
  | 'selfGeneration'      // 模式1：自生成组合输出
  | 'selfPlay'           // 模式2：自博弈竞争优化
  | 'selfEvaluation';    // 模式3：自双输出+自评判

/** 候选结果 */
export interface CandidateResult {
  id: string;
  content: string;
  score: number;
  weight: number;
  stdpWeight: number;
}

/** 评判维度 */
export interface EvaluationDimension {
  name: 'factAccuracy' | 'logicCompleteness' | 'semanticCoherence' | 'instructionFollowing';
  score: number;  // 0-10分
  weight: number;
}

/** 评判结果 */
export interface EvaluationResult {
  candidateId: string;
  dimensions: EvaluationDimension[];
  totalScore: number;  // 总分40分
  feedback: string;
}

// ==================== 推理引擎类型 ====================

/** 单周期执行状态 */
export interface CycleExecutionState {
  cycleId: number;
  startTime: number;
  endTime: number;
  phase: CyclePhase;
  inputToken: TokenFeature | null;
  memoryAnchors: MemoryAnchor[];
  output: string;
  stdpUpdates: STDPUpdateRecord[];
}

/** 周期执行阶段 */
export type CyclePhase = 
  | 'inputReceive'        // 1.输入token接收与特征提取
  | 'memoryRecall'        // 2.海马体记忆锚点调取
  | 'inference'           // 3.窄窗口上下文推理
  | 'output'              // 4.单周期输出结果生成
  | 'stdpUpdate'          // 5.全链路STDP权重刷新
  | 'memoryEncode'        // 6.海马体情景记忆编码
  | 'workingMemoryUpdate' // 7.全局工作记忆压缩更新
  | 'complete';           // 周期完成

/** 推理引擎配置 */
export interface InferenceEngineConfig {
  refreshRateHz: 100;  // 刷新率
  cycleDurationMs: 10;  // 周期时长
  narrowWindowSize: 1 | 2;  // 窄窗口大小
  maxMemoryAnchors: 2;  // 单周期最大记忆锚点数
  attentionComplexity: 'O(1)';  // 注意力复杂度
}

// ==================== 模型架构类型 ====================

/** 权重分支 */
export interface WeightBranch {
  type: 'static' | 'dynamic';
  ratio: number;
  weights: Map<string, Float32Array>;
  frozen: boolean;  // 是否冻结
}

/** 注意力层改造 */
export interface AttentionLayerModified {
  layerId: string;
  staticBranch: WeightBranch;   // 90%静态分支
  dynamicBranch: WeightBranch;  // 10%动态分支
  featureOutputInterface: (token: TokenFeature) => TokenFeature;
  hippocampusGateInterface: (anchors: MemoryAnchor[]) => void;
}

/** FFN层改造 */
export interface FFNLayerModified {
  layerId: string;
  staticBranch: WeightBranch;
  dynamicBranch: WeightBranch;
}

/** 模型改造配置 */
export interface ModelModificationConfig {
  baseModel: 'Qwen3.5-0.8B';
  staticWeightRatio: 0.9;
  dynamicWeightRatio: 0.1;
  attentionLayers: AttentionLayerModified[];
  ffnLayers: FFNLayerModified[];
  outputLayer: WeightBranch[];
}

// ==================== 训练类型 ====================

/** 训练配置 */
export interface TrainingConfig {
  targetWeights: 'dynamicOnly';  // 仅训练动态权重
  method: 'STDP';  // STDP规则
  epochs: number;
  batchSize: number;
  learningRate: number;
}

/** 训练记录 */
export interface TrainingRecord {
  epoch: number;
  step: number;
  loss: number;
  stdpUpdates: STDPUpdateRecord[];
  memoryAnchorsCreated: number;
  evaluationScore: number;
}

// ==================== 测评类型 ====================

/** 测评维度 */
export interface EvaluationMetrics {
  inferenceLatency: number;  // 推理延迟(ms)
  memoryUsage: number;  // 内存占用(MB)
  accuracy: number;  // 准确率
  coherence: number;  // 连贯性
  learningEfficiency: number;  // 学习效率
  memoryRecallAccuracy: number;  // 记忆召回准确率
  stdpUpdateCount: number;  // STDP更新次数
  cycleCompletionRate: number;  // 周期完成率
}

/** 测评报告 */
export interface EvaluationReport {
  timestamp: number;
  metrics: EvaluationMetrics;
  passed: boolean;
  details: Record<string, unknown>;
}

// ==================== 端侧部署类型 ====================

/** 端侧设备类型 */
export type EdgeDevice = 'android' | 'raspberryPi4' | 'raspberryPi5' | 'jetsonNano';

/** 端侧部署配置 */
export interface EdgeDeploymentConfig {
  device: EdgeDevice;
  quantization: QuantizationConfig;
  maxMemoryMB: number;
  targetLatencyMs: number;
  offlineMode: boolean;
}

// ==================== 全局状态类型 ====================

/** 全局工作记忆 */
export interface WorkingMemory {
  currentContext: NarrowWindowContext;
  recentOutputs: string[];
  activeMemoryAnchors: MemoryAnchor[];
  cycleHistory: CycleExecutionState[];
  maxHistoryLength: number;
}

/** 架构全局状态 */
export interface ArchitectureState {
  isRunning: boolean;
  currentCycle: number;
  currentPhase: CyclePhase;
  workingMemory: WorkingMemory;
  stdpWeights: Map<string, STDPWeight>;
  hippocampus: HippocampusModule;
  optimizationMode: OptimizationMode;
  deviceConfig: EdgeDeploymentConfig;
}
