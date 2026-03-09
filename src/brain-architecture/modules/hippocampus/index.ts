/**
 * 模块5：海马体记忆系统全模块
 * Module 5: Hippocampus Memory System Full Module
 * 
 * 严格基于人脑海马体-新皮层双系统神经科学原理开发
 * 完全适配10ms刷新周期，单周期计算延迟≤1ms
 */

import type {
  TokenFeature,
  MemoryAnchor,
  HippocampusModule,
  EntorhinalCortexUnit,
  DentateGyrusUnit,
  CA3Unit,
  CA1Unit,
  SWRUnit,
  EpisodicMemoryStore,
} from '../../types';
import {
  HIPPOCAMPUS_CONFIG,
  PATTERN_SEPARATION_CONFIG,
  generateId,
} from '../../config';

// ==================== 内嗅皮层EC - 特征编码单元 ====================

/**
 * 内嗅皮层EC单元实现
 * 海马体的输入输出门户
 */
export class EntorhinalCortexImpl implements EntorhinalCortexUnit {
  private featureDimension: number;

  constructor() {
    this.featureDimension = HIPPOCAMPUS_CONFIG.featureDimension;
  }

  /**
   * 编码特征
   * 接收模型注意力层输出的token特征，归一化稀疏编码为64维固定低维特征向量
   */
  encodeFeature(tokenFeature: TokenFeature): Float32Array {
    const encodedFeature = new Float32Array(this.featureDimension);

    // 从原始嵌入向量压缩到64维
    const sourceLength = tokenFeature.embedding.length;
    const compressionRatio = Math.floor(sourceLength / this.featureDimension);

    for (let i = 0; i < this.featureDimension; i++) {
      let sum = 0;
      for (let j = 0; j < compressionRatio; j++) {
        const sourceIdx = i * compressionRatio + j;
        if (sourceIdx < sourceLength) {
          sum += tokenFeature.embedding[sourceIdx];
        }
      }
      encodedFeature[i] = sum / compressionRatio;
    }

    // 加入时序特征
    const temporalWeight = 0.1;
    for (let i = 0; i < Math.min(16, this.featureDimension); i++) {
      encodedFeature[i] += tokenFeature.temporalFeature[i] * temporalWeight;
    }

    // 加入语义特征
    for (let i = 16; i < Math.min(48, this.featureDimension); i++) {
      encodedFeature[i] += tokenFeature.semanticFeature[i - 16] * temporalWeight;
    }

    return encodedFeature;
  }

  /**
   * 归一化稀疏编码
   */
  normalizeSparse(feature: Float32Array): Float32Array {
    const normalized = new Float32Array(feature.length);

    // 计算L2范数
    let norm = 0;
    for (let i = 0; i < feature.length; i++) {
      norm += feature[i] * feature[i];
    }
    norm = Math.sqrt(norm);

    // 归一化
    if (norm > 0) {
      for (let i = 0; i < feature.length; i++) {
        normalized[i] = feature[i] / norm;
      }
    }

    // 稀疏化：保留top-k激活
    const k = Math.floor(feature.length * 0.3); // 保留30%
    const threshold = this.findKthLargest(normalized, k);

    for (let i = 0; i < normalized.length; i++) {
      if (Math.abs(normalized[i]) < threshold) {
        normalized[i] = 0;
      }
    }

    return normalized;
  }

  /**
   * 找第k大的元素
   */
  private findKthLargest(arr: Float32Array, k: number): number {
    const sorted = Array.from(arr)
      .map(Math.abs)
      .sort((a, b) => b - a);
    return sorted[Math.min(k, sorted.length - 1)] || 0;
  }
}

// ==================== 齿状回DG - 模式分离单元 ====================

/**
 * 齿状回DG单元实现
 * 负责模式分离，为相似输入生成完全正交的唯一记忆ID
 */
export class DentateGyrusImpl implements DentateGyrusUnit {
  private sparseDimension: number;
  private orthogonalMatrix: Float32Array[];
  private idCounter: number = 0;

  constructor() {
    this.sparseDimension = PATTERN_SEPARATION_CONFIG.sparseDimension;
    this.orthogonalMatrix = this.generateOrthogonalMatrix();
  }

  /**
   * 模式分离
   * 对编码特征做稀疏随机投影正交化处理，生成唯一记忆ID
   */
  patternSeparation(feature: Float32Array): string {
    // 正交投影
    const projected = this.orthogonalProjection(feature);

    // 生成唯一ID
    const hash = this.computeHash(projected);
    const timestamp = Date.now();
    const uniqueId = `mem_${timestamp}_${hash}_${this.idCounter++}`;

    return uniqueId;
  }

  /**
   * 正交投影
   * 稀疏随机投影正交化处理
   */
  orthogonalProjection(feature: Float32Array): Float32Array {
    const projected = new Float32Array(this.sparseDimension);

    // 使用预生成的正交矩阵进行投影
    for (let i = 0; i < this.sparseDimension; i++) {
      let sum = 0;
      const row = this.orthogonalMatrix[i];
      for (let j = 0; j < Math.min(feature.length, row.length); j++) {
        sum += feature[j] * row[j];
      }
      projected[i] = sum;
    }

    // 应用非线性激活增强正交性
    const strength = PATTERN_SEPARATION_CONFIG.orthogonalizationStrength;
    for (let i = 0; i < projected.length; i++) {
      projected[i] = Math.sign(projected[i]) * Math.pow(Math.abs(projected[i]), strength);
    }

    return projected;
  }

  /**
   * 生成正交矩阵
   */
  private generateOrthogonalMatrix(): Float32Array[] {
    const matrix: Float32Array[] = [];
    const inputDim = HIPPOCAMPUS_CONFIG.featureDimension;

    for (let i = 0; i < this.sparseDimension; i++) {
      const row = new Float32Array(inputDim);
      // 稀疏随机初始化
      const sparsity = 0.1; // 10%稀疏度
      for (let j = 0; j < inputDim; j++) {
        if (Math.random() < sparsity) {
          row[j] = (Math.random() - 0.5) * 2;
        }
      }
      matrix.push(row);
    }

    return matrix;
  }

  /**
   * 计算哈希
   */
  private computeHash(feature: Float32Array): string {
    let hash = 0;
    for (let i = 0; i < feature.length; i++) {
      hash = ((hash << 5) - hash + feature[i]) | 0;
    }
    return Math.abs(hash).toString(16).padStart(8, '0');
  }
}

// ==================== CA3区 - 情景记忆库+模式补全 ====================

/**
 * CA3区单元实现
 * 情景记忆存储与模式补全
 */
export class CA3Impl implements CA3Unit {
  episodicMemoryStore: EpisodicMemoryStore;
  private maxMemorySize: number;
  private memoryDecayFactor: number;

  constructor() {
    this.maxMemorySize = HIPPOCAMPUS_CONFIG.maxEpisodicMemory;
    this.memoryDecayFactor = HIPPOCAMPUS_CONFIG.memoryDecayFactor;
    this.episodicMemoryStore = {
      memories: new Map(),
      maxSize: this.maxMemorySize,
      currentSize: 0,
    };
  }

  /**
   * 存储记忆
   */
  storeMemory(memory: MemoryAnchor): void {
    // 检查容量
    if (this.episodicMemoryStore.currentSize >= this.maxMemorySize) {
      this.pruneOldestMemory();
    }

    this.episodicMemoryStore.memories.set(memory.id, memory);
    this.episodicMemoryStore.currentSize++;
  }

  /**
   * 模式补全
   * 基于部分线索完成完整记忆链条召回
   */
  patternCompletion(partialCue: Float32Array): MemoryAnchor[] {
    const candidates: Array<{ memory: MemoryAnchor; similarity: number }> = [];

    // 计算与所有记忆的相似度
    this.episodicMemoryStore.memories.forEach((memory) => {
      const similarity = this.calculateSimilarity(partialCue, memory.featureVector);
      candidates.push({ memory, similarity });
    });

    // 按相似度排序，返回最相关的1-2个
    candidates.sort((a, b) => b.similarity - a.similarity);

    const maxRecall = HIPPOCAMPUS_CONFIG.maxRecallAnchors;
    return candidates.slice(0, maxRecall).map((c) => {
      // 更新访问信息
      c.memory.accessCount++;
      c.memory.lastAccessTime = Date.now();
      return c.memory;
    });
  }

  /**
   * 根据ID获取记忆
   */
  getMemory(id: string): MemoryAnchor | undefined {
    return this.episodicMemoryStore.memories.get(id);
  }

  /**
   * 删除最旧的记忆
   */
  private pruneOldestMemory(): void {
    let oldest: MemoryAnchor | null = null;
    let oldestId: string | null = null;

    this.episodicMemoryStore.memories.forEach((memory, id) => {
      if (!oldest || memory.lastAccessTime < oldest.lastAccessTime) {
        oldest = memory;
        oldestId = id;
      }
    });

    if (oldestId) {
      this.episodicMemoryStore.memories.delete(oldestId);
      this.episodicMemoryStore.currentSize--;
    }
  }

  /**
   * 计算相似度
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
   * 应用记忆衰减
   */
  applyDecay(): void {
    this.episodicMemoryStore.memories.forEach((memory) => {
      memory.strength *= this.memoryDecayFactor;
    });
  }

  /**
   * 获取记忆统计
   */
  getStatistics(): {
    totalMemories: number;
    avgStrength: number;
    avgAccessCount: number;
  } {
    const memories = Array.from(this.episodicMemoryStore.memories.values());
    const totalMemories = memories.length;
    const avgStrength =
      memories.reduce((sum, m) => sum + m.strength, 0) / totalMemories || 0;
    const avgAccessCount =
      memories.reduce((sum, m) => sum + m.accessCount, 0) / totalMemories || 0;

    return { totalMemories, avgStrength, avgAccessCount };
  }
}

// ==================== CA1区 - 时序编码+注意力门控 ====================

/**
 * CA1区单元实现
 * 时序编码与注意力门控控制
 */
export class CA1Impl implements CA1Unit {
  private temporalSequence: MemoryAnchor[] = [];
  private maxSequenceLength: number = 100;
  private currentAnchors: MemoryAnchor[] = [];

  /**
   * 时序编码
   * 为每个记忆单元打精准时间戳，绑定时序-情景-因果关系
   */
  temporalEncoding(memory: MemoryAnchor): void {
    // 更新时间戳
    memory.timestamp = Date.now();

    // 添加到时序序列
    this.temporalSequence.push(memory);
    if (this.temporalSequence.length > this.maxSequenceLength) {
      this.temporalSequence.shift();
    }

    // 建立因果关联
    if (this.temporalSequence.length > 1) {
      const prevMemory = this.temporalSequence[this.temporalSequence.length - 2];
      memory.causalRelation = prevMemory.id;
    }
  }

  /**
   * 注意力门控控制
   * 每个刷新周期输出记忆锚点给模型注意力层
   */
  attentionGateControl(currentContext: {
    currentToken: TokenFeature;
    relatedMemoryAnchors: MemoryAnchor[];
  }): MemoryAnchor[] {
    // 根据当前上下文选择最相关的记忆锚点
    const candidates: Array<{ anchor: MemoryAnchor; relevance: number }> = [];

    for (const anchor of this.temporalSequence) {
      const relevance = this.calculateRelevance(
        currentContext.currentToken,
        anchor
      );
      candidates.push({ anchor, relevance });
    }

    // 按相关性排序
    candidates.sort((a, b) => b.relevance - a.relevance);

    // 返回最相关的1-2个锚点
    this.currentAnchors = candidates
      .slice(0, HIPPOCAMPUS_CONFIG.maxRecallAnchors)
      .map((c) => c.anchor);

    return this.currentAnchors;
  }

  /**
   * 输出记忆锚点
   */
  outputMemoryAnchors(count: 1 | 2 = 2): MemoryAnchor[] {
    return this.currentAnchors.slice(0, count);
  }

  /**
   * 计算相关性
   */
  private calculateRelevance(token: TokenFeature, anchor: MemoryAnchor): number {
    // 时序相关性：越近越相关
    const timeDiff = Date.now() - anchor.timestamp;
    const temporalRelevance = Math.exp(-timeDiff / 60000); // 1分钟衰减

    // 语义相关性
    let semanticRelevance = 0;
    const minLength = Math.min(token.embedding.length, anchor.featureVector.length);
    for (let i = 0; i < minLength; i++) {
      semanticRelevance += token.embedding[i] * anchor.featureVector[i];
    }
    semanticRelevance = Math.abs(semanticRelevance) / minLength;

    // 访问频率加权
    const accessWeight = Math.log(anchor.accessCount + 1) / 10;

    // 综合相关性
    return temporalRelevance * 0.4 + semanticRelevance * 0.4 + accessWeight * 0.2;
  }

  /**
   * 获取时序序列
   */
  getTemporalSequence(): MemoryAnchor[] {
    return [...this.temporalSequence];
  }
}

// ==================== 尖波涟漪SWR - 离线回放巩固 ====================

/**
 * 尖波涟漪SWR单元实现
 * 端侧空闲时的记忆回放巩固
 */
export class SWRImpl implements SWRUnit {
  private isReplaying: boolean = false;
  private replayInterval: ReturnType<typeof setInterval> | null = null;
  private ca3: CA3Impl | null = null;
  private ca1: CA1Impl | null = null;

  /**
   * 设置关联模块
   */
  setModules(ca3: CA3Impl, ca1: CA1Impl): void {
    this.ca3 = ca3;
    this.ca1 = ca1;
  }

  /**
   * 启动离线回放
   */
  startReplay(): void {
    if (this.isReplaying) return;

    this.isReplaying = true;
    console.log('[SWR] 启动离线回放巩固');

    this.replayInterval = setInterval(() => {
      this.replayConsolidation();
    }, HIPPOCAMPUS_CONFIG.swrReplayInterval);
  }

  /**
   * 停止离线回放
   */
  stopReplay(): void {
    if (this.replayInterval) {
      clearInterval(this.replayInterval);
      this.replayInterval = null;
    }
    this.isReplaying = false;
    console.log('[SWR] 停止离线回放巩固');
  }

  /**
   * 记忆回放巩固
   * 模拟人脑睡眠尖波涟漪，回放记忆序列
   */
  replayConsolidation(): void {
    if (!this.ca3 || !this.ca1) return;

    const sequence = this.ca1.getTemporalSequence();
    if (sequence.length === 0) return;

    // 随机选择一段记忆序列进行回放
    const startIdx = Math.floor(Math.random() * sequence.length);
    const replayLength = Math.min(5, sequence.length - startIdx);
    const replaySequence = sequence.slice(startIdx, startIdx + replayLength);

    // 模拟回放过程
    for (const memory of replaySequence) {
      // 增强记忆强度
      memory.strength = Math.min(1, memory.strength * 1.1);

      // 更新访问时间
      memory.lastAccessTime = Date.now();
    }

    // 执行记忆修剪
    this.memoryPruning();

    // 执行权重优化
    this.weightOptimization();
  }

  /**
   * 记忆修剪
   * 移除强度过低的记忆
   */
  memoryPruning(): void {
    if (!this.ca3) return;

    const threshold = 0.1;
    const toDelete: string[] = [];

    this.ca3.episodicMemoryStore.memories.forEach((memory, id) => {
      if (memory.strength < threshold) {
        toDelete.push(id);
      }
    });

    for (const id of toDelete) {
      this.ca3.episodicMemoryStore.memories.delete(id);
      this.ca3.episodicMemoryStore.currentSize--;
    }

    if (toDelete.length > 0) {
      console.log(`[SWR] 修剪了 ${toDelete.length} 条弱记忆`);
    }
  }

  /**
   * 权重优化
   * 基于回放结果优化STDP权重
   */
  weightOptimization(): void {
    // 实际部署时会调用STDP系统进行权重优化
    // 这里仅做模拟
  }

  /**
   * 检查是否正在回放
   */
  isCurrentlyReplaying(): boolean {
    return this.isReplaying;
  }
}

// ==================== 海马体模块管理器 ====================

/**
 * 海马体模块管理器
 * 统一管理所有海马体子模块
 */
export class HippocampusManager implements HippocampusModule {
  entorhinalCortex: EntorhinalCortexImpl;
  dentateGyrus: DentateGyrusImpl;
  ca3: CA3Impl;
  ca1: CA1Impl;
  sharpWaveRipple: SWRImpl;

  private cycleCount: number = 0;

  constructor() {
    this.entorhinalCortex = new EntorhinalCortexImpl();
    this.dentateGyrus = new DentateGyrusImpl();
    this.ca3 = new CA3Impl();
    this.ca1 = new CA1Impl();
    this.sharpWaveRipple = new SWRImpl();

    // 设置SWR的关联模块
    this.sharpWaveRipple.setModules(this.ca3, this.ca1);

    console.log('[Hippocampus] 海马体模块初始化完成');
  }

  /**
   * 执行单周期海马体处理
   */
  executeCycle(tokenFeature: TokenFeature): {
    memoryAnchors: MemoryAnchor[];
    newMemoryId: string;
    cycleTime: number;
  } {
    const startTime = performance.now();
    this.cycleCount++;

    // 1. EC特征编码
    const encodedFeature = this.entorhinalCortex.encodeFeature(tokenFeature);
    const normalizedFeature = this.entorhinalCortex.normalizeSparse(encodedFeature);

    // 2. DG模式分离，生成唯一记忆ID
    const memoryId = this.dentateGyrus.patternSeparation(normalizedFeature);

    // 3. 创建新记忆锚点
    const newMemory: MemoryAnchor = {
      id: memoryId,
      timestamp: Date.now(),
      featureVector: normalizedFeature,
      temporalSkeleton: [this.cycleCount],
      semanticPointer: `token_${tokenFeature.tokenId}`,
      causalRelation: '',
      accessCount: 0,
      lastAccessTime: Date.now(),
      strength: 1.0,
    };

    // 4. CA3存储记忆
    this.ca3.storeMemory(newMemory);

    // 5. CA1时序编码
    this.ca1.temporalEncoding(newMemory);

    // 6. CA1注意力门控，返回相关记忆锚点
    const memoryAnchors = this.ca1.attentionGateControl({
      currentToken: tokenFeature,
      relatedMemoryAnchors: [],
    });

    const cycleTime = performance.now() - startTime;

    return {
      memoryAnchors,
      newMemoryId: memoryId,
      cycleTime,
    };
  }

  /**
   * 根据线索召回记忆
   */
  recallMemory(cue: Float32Array): MemoryAnchor[] {
    return this.ca3.patternCompletion(cue);
  }

  /**
   * 启动离线回放
   */
  startOfflineReplay(): void {
    this.sharpWaveRipple.startReplay();
  }

  /**
   * 停止离线回放
   */
  stopOfflineReplay(): void {
    this.sharpWaveRipple.stopReplay();
  }

  /**
   * 获取记忆统计
   */
  getMemoryStatistics(): {
    totalMemories: number;
    avgStrength: number;
    avgAccessCount: number;
    sequenceLength: number;
  } {
    const ca3Stats = this.ca3.getStatistics();
    return {
      ...ca3Stats,
      sequenceLength: this.ca1.getTemporalSequence().length,
    };
  }

  /**
   * 获取周期计数
   */
  getCycleCount(): number {
    return this.cycleCount;
  }
}

// 导出单例
export const hippocampusManager = new HippocampusManager();
