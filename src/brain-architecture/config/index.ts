/**
 * 类人脑双系统全闭环AI架构 - 全局配置
 * Human-Like Brain Dual-System Full-Loop AI Architecture - Global Configuration
 */

import type {
  STDPConfig,
  InferenceEngineConfig,
  QuantizationConfig,
  WeightSplitRatio,
  EdgeDeploymentConfig,
} from '../types';

// ==================== 核心常量 ====================

/** 刷新周期常量 */
export const REFRESH_CYCLE_MS = 10; // 10ms = 100Hz
export const REFRESH_RATE_HZ = 100;

/** 权重拆分比例 */
export const WEIGHT_SPLIT_RATIO: WeightSplitRatio = {
  static: 0.9,   // 90%静态基础权重
  dynamic: 0.1,  // 10%STDP动态增量权重
};

/** 量化配置 */
export const QUANTIZATION_CONFIG: QuantizationConfig = {
  precision: 'INT4',
  maxMemoryMB: 420,
  maxComputeOverhead: 0.1,
};

// ==================== STDP配置 ====================

/** 默认STDP配置 */
export const DEFAULT_STDP_CONFIG: STDPConfig = {
  alpha: 0.01,        // LTP学习率
  beta: 0.005,        // LTD学习率
  updateThreshold: 0.001,
  weightUpperBound: 1.0,
  weightLowerBound: -1.0,
};

/** STDP更新频率配置 */
export const STDP_UPDATE_INTERVALS = {
  attention: 1,        // 每个刷新周期
  ffn: 1,              // 每个刷新周期
  selfEvaluation: 10,  // 每10个刷新周期
  hippocampusGate: 1,  // 每个刷新周期
};

// ==================== 推理引擎配置 ====================

/** 默认推理引擎配置 */
export const DEFAULT_INFERENCE_CONFIG: InferenceEngineConfig = {
  refreshRateHz: 100,
  cycleDurationMs: 10,
  narrowWindowSize: 2,
  maxMemoryAnchors: 2,
  attentionComplexity: 'O(1)',
};

/** 单周期执行阶段时间分配(ms) */
export const CYCLE_PHASE_TIMING = {
  inputReceive: 0.5,
  memoryRecall: 1.0,
  inference: 5.0,
  output: 1.0,
  stdpUpdate: 1.5,
  memoryEncode: 0.5,
  workingMemoryUpdate: 0.5,
};

// ==================== 海马体配置 ====================

/** 海马体模块配置 */
export const HIPPOCAMPUS_CONFIG = {
  /** 特征向量维度 */
  featureDimension: 64,
  /** 情景记忆库最大容量 */
  maxEpisodicMemory: 10000,
  /** 情景记忆库最大内存占用(MB) */
  maxMemoryMB: 2,
  /** 记忆锚点召回数量 */
  maxRecallAnchors: 2,
  /** 记忆强度衰减因子 */
  memoryDecayFactor: 0.99,
  /** SWR回放频率(空闲时) */
  swrReplayInterval: 1000, // ms
};

/** 模式分离配置 */
export const PATTERN_SEPARATION_CONFIG = {
  /** 稀疏投影维度 */
  sparseDimension: 128,
  /** 正交化强度 */
  orthogonalizationStrength: 0.8,
};

// ==================== 自闭环优化配置 ====================

/** 模式切换关键词 */
export const MODE_SWITCH_KEYWORDS = {
  selfPlay: [
    '计算', '推理', '逻辑', '数学', '代码', '编程',
    '证明', '分析', '推导', '算法',
  ],
  selfEvaluation: [
    '方案', '建议', '决策', '评估', '选择', '最优',
    '专业', '准确', '完整',
  ],
};

/** 评判维度权重 */
export const EVALUATION_DIMENSION_WEIGHTS = {
  factAccuracy: 0.3,
  logicCompleteness: 0.25,
  semanticCoherence: 0.25,
  instructionFollowing: 0.2,
};

/** 自博弈最大迭代次数 */
export const SELF_PLAY_MAX_ITERATIONS = 5;

// ==================== 端侧部署配置 ====================

/** 各设备默认配置 */
export const EDGE_DEVICE_CONFIGS: Record<string, EdgeDeploymentConfig> = {
  android: {
    device: 'android',
    quantization: QUANTIZATION_CONFIG,
    maxMemoryMB: 420,
    targetLatencyMs: 10,
    offlineMode: true,
  },
  raspberryPi4: {
    device: 'raspberryPi4',
    quantization: QUANTIZATION_CONFIG,
    maxMemoryMB: 420,
    targetLatencyMs: 15,
    offlineMode: true,
  },
  raspberryPi5: {
    device: 'raspberryPi5',
    quantization: QUANTIZATION_CONFIG,
    maxMemoryMB: 500,
    targetLatencyMs: 10,
    offlineMode: true,
  },
  jetsonNano: {
    device: 'jetsonNano',
    quantization: QUANTIZATION_CONFIG,
    maxMemoryMB: 600,
    targetLatencyMs: 8,
    offlineMode: true,
  },
};

// ==================== 训练配置 ====================

/** 预适配微调配置 */
export const PRE_ADAPTATION_CONFIG = {
  epochs: 3,
  batchSize: 8,
  learningRate: 0.001,
  targetLayers: ['dynamicBranch', 'hippocampusConnections'],
};

/** 在线学习配置 */
export const ONLINE_LEARNING_CONFIG = {
  enableRealTimeUpdate: true,
  stdpOnly: true,
  noBackprop: true,
  updateInterval: REFRESH_CYCLE_MS,
};

// ==================== 测评配置 ====================

/** 测评基准 */
export const EVALUATION_BENCHMARKS = {
  inferenceLatency: { target: 10, unit: 'ms', max: 15 },
  memoryUsage: { target: 400, unit: 'MB', max: 420 },
  accuracy: { target: 0.85, unit: 'ratio', min: 0.8 },
  coherence: { target: 0.9, unit: 'ratio', min: 0.85 },
  learningEfficiency: { target: 0.8, unit: 'ratio', min: 0.7 },
  memoryRecallAccuracy: { target: 0.9, unit: 'ratio', min: 0.85 },
  cycleCompletionRate: { target: 0.99, unit: 'ratio', min: 0.95 },
};

// ==================== 角色提示词模板 ====================

/** 角色切换提示词模板 */
export const ROLE_PROMPTS = {
  generator: `你是一个智能生成助手，负责根据输入生成高质量、连贯的回复。请确保输出准确、有逻辑、符合用户需求。`,
  
  validator: `你是一个严谨的验证助手，负责对生成的结果进行逻辑校验、事实核查和漏洞排查。请仔细检查推理过程，指出任何错误或不一致之处。`,
  
  evaluator: `你是一个专业的评判助手，负责对候选结果进行多维度评分。
评分维度：
1. 事实准确性(0-10分)：内容是否准确无误
2. 逻辑完整性(0-10分)：推理是否完整、无逻辑漏洞
3. 语义连贯性(0-10分)：表达是否流畅、连贯
4. 指令遵循度(0-10分)：是否完全遵循用户指令

请严格按照以上维度打分，总分40分。`,
};

// ==================== 模型配置 ====================

/** Qwen3.5-0.8B模型配置 */
export const QWEN_MODEL_CONFIG = {
  modelName: 'Qwen3.5-0.8B',
  hiddenSize: 896,
  numLayers: 24,
  numAttentionHeads: 14,
  intermediateSize: 4864,
  vocabSize: 151936,
  maxPositionEmbeddings: 32768,
};

// ==================== 工具函数 ====================

/** 获取当前时间戳(ms) */
export function getCurrentTimestamp(): number {
  return Date.now();
}

/** 生成唯一ID */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/** 延迟函数 */
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/** 检查是否在端侧环境 */
export function isEdgeEnvironment(): boolean {
  if (typeof window !== 'undefined') {
    return true;
  }
  return false;
}

/** 计算内存占用估算 */
export function estimateMemoryUsage(
  modelSize: number,
  quantization: 'INT4' | 'INT8' | 'FP16' | 'FP32'
): number {
  const quantizationFactors = {
    INT4: 0.125,
    INT8: 0.25,
    FP16: 0.5,
    FP32: 1.0,
  };
  return modelSize * quantizationFactors[quantization];
}
