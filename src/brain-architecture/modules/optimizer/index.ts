/**
 * 模块4：单智体自生成-自博弈-自评判闭环优化系统
 * Module 4: Single Agent Self-Generation-Self-Play-Self-Evaluation Closed-Loop Optimization System
 * 
 * 实现单模型内的组合输出、竞争优化、自双输出+自评判全能力
 */

import type {
  OptimizationMode,
  CandidateResult,
  EvaluationDimension,
  EvaluationResult,
  TokenFeature,
  MemoryAnchor,
  STDPUpdateRecord,
} from '../../types';
import {
  MODE_SWITCH_KEYWORDS,
  EVALUATION_DIMENSION_WEIGHTS,
  SELF_PLAY_MAX_ITERATIONS,
  ROLE_PROMPTS,
  generateId,
} from '../../config';
import { stdpSystem } from '../stdp';

// ==================== 模式切换器 ====================

/**
 * 模式自动切换器
 * 根据输入关键词、任务难度自动切换模式
 */
export class ModeSwitcher {
  /**
   * 分析输入并决定运行模式
   */
  analyzeAndSwitch(input: string): OptimizationMode {
    const lowerInput = input.toLowerCase();

    // 检查自博弈关键词
    for (const keyword of MODE_SWITCH_KEYWORDS.selfPlay) {
      if (lowerInput.includes(keyword)) {
        return 'selfPlay';
      }
    }

    // 检查自评判关键词
    for (const keyword of MODE_SWITCH_KEYWORDS.selfEvaluation) {
      if (lowerInput.includes(keyword)) {
        return 'selfEvaluation';
      }
    }

    // 默认：自生成组合输出
    return 'selfGeneration';
  }

  /**
   * 获取模式描述
   */
  getModeDescription(mode: OptimizationMode): string {
    const descriptions: Record<OptimizationMode, string> = {
      selfGeneration: '自生成组合输出模式：并行生成候选，STDP加权投票',
      selfPlay: '自博弈竞争优化模式：提案-验证对抗迭代',
      selfEvaluation: '自双输出+自评判模式：双候选生成+多维度评判',
    };
    return descriptions[mode];
  }
}

// ==================== 模式1：自生成组合输出 ====================

/**
 * 自生成组合输出执行器
 * 每个刷新周期并行生成2个候选，STDP加权投票
 */
export class SelfGenerationExecutor {
  private candidateHistory: CandidateResult[] = [];
  private accuracyHistory: number[] = [];
  private maxHistoryLength: number = 10;

  /**
   * 执行自生成组合输出
   */
  execute(
    input: string,
    currentCycle: number
  ): {
    result: string;
    candidates: CandidateResult[];
    stdpUpdates: STDPUpdateRecord[];
  } {
    // 并行生成2个候选
    const candidate1 = this.generateCandidate(input, 0.7, currentCycle);
    const candidate2 = this.generateCandidate(input, 0.9, currentCycle);

    // 计算STDP加权权重
    const weight1 = this.calculateSTDPWeight(candidate1);
    const weight2 = this.calculateSTDPWeight(candidate2);

    candidate1.weight = weight1;
    candidate2.weight = weight2;

    // 一致性投票
    const result = this.voteOnCandidates([candidate1, candidate2]);

    // 更新历史
    this.candidateHistory.push(candidate1, candidate2);
    if (this.candidateHistory.length > this.maxHistoryLength * 2) {
      this.candidateHistory = this.candidateHistory.slice(-this.maxHistoryLength * 2);
    }

    // 生成STDP更新
    const stdpUpdates = this.generateSTDPUpdates([candidate1, candidate2], result);

    return {
      result,
      candidates: [candidate1, candidate2],
      stdpUpdates,
    };
  }

  /**
   * 生成候选结果
   */
  private generateCandidate(
    input: string,
    temperature: number,
    cycle: number
  ): CandidateResult {
    // 模拟生成过程
    const baseScore = 0.5 + Math.random() * 0.5;
    const variance = (1 - temperature) * 0.3;

    return {
      id: generateId(),
      content: `候选结果_${cycle}_${temperature.toFixed(1)}`,
      score: baseScore + (Math.random() - 0.5) * variance,
      weight: 1,
      stdpWeight: 0.5,
    };
  }

  /**
   * 计算STDP权重
   * 基于过往准确率
   */
  private calculateSTDPWeight(candidate: CandidateResult): number {
    if (this.accuracyHistory.length === 0) {
      return 0.5;
    }

    // 计算平均准确率
    const avgAccuracy =
      this.accuracyHistory.reduce((a, b) => a + b, 0) / this.accuracyHistory.length;

    // 根据候选分数和准确率调整权重
    return candidate.score * avgAccuracy;
  }

  /**
   * 候选投票
   */
  private voteOnCandidates(candidates: CandidateResult[]): string {
    // 加权投票
    const totalWeight = candidates.reduce((sum, c) => sum + c.weight, 0);

    if (totalWeight === 0) {
      return candidates[0].content;
    }

    // 选择权重最高的候选
    const sorted = [...candidates].sort((a, b) => b.weight - a.weight);
    return sorted[0].content;
  }

  /**
   * 生成STDP更新
   */
  private generateSTDPUpdates(
    candidates: CandidateResult[],
    selectedResult: string
  ): STDPUpdateRecord[] {
    const updates: STDPUpdateRecord[] = [];
    const now = Date.now();

    for (const candidate of candidates) {
      const isSelected = candidate.content === selectedResult;
      const deltaValue = isSelected ? 0.01 : -0.005;

      updates.push({
        timestamp: now,
        layerType: 'selfEvaluation',
        weightId: candidate.id,
        deltaValue,
        updateType: isSelected ? 'LTP' : 'LTD',
        triggerReason: isSelected ? '正确路径增强' : '错误路径减弱',
      });
    }

    return updates;
  }

  /**
   * 更新准确率历史
   */
  updateAccuracy(accuracy: number): void {
    this.accuracyHistory.push(accuracy);
    if (this.accuracyHistory.length > this.maxHistoryLength) {
      this.accuracyHistory.shift();
    }
  }
}

// ==================== 模式2：自博弈竞争优化 ====================

/**
 * 自博弈竞争优化执行器
 * 提案角色-验证角色对抗迭代
 */
export class SelfPlayExecutor {
  private iterationCount: number = 0;
  private maxIterations: number = SELF_PLAY_MAX_ITERATIONS;

  /**
   * 执行自博弈竞争优化
   */
  execute(
    input: string,
    onIteration?: (iteration: number, proposal: string, validation: string) => void
  ): {
    result: string;
    iterations: Array<{ proposal: string; validation: string; converged: boolean }>;
    stdpUpdates: STDPUpdateRecord[];
  } {
    const iterations: Array<{
      proposal: string;
      validation: string;
      converged: boolean;
    }> = [];
    const stdpUpdates: STDPUpdateRecord[] = [];
    let currentProposal = input;
    let converged = false;

    for (let i = 0; i < this.maxIterations && !converged; i++) {
      this.iterationCount++;

      // 奇数周期：提案角色生成
      const proposal = this.generateProposal(currentProposal, i);

      // 偶数周期：验证角色校验
      const validation = this.validateProposal(proposal, i);

      // 检查是否收敛
      converged = this.checkConvergence(proposal, validation);

      iterations.push({
        proposal,
        validation,
        converged,
      });

      // 更新当前提案
      currentProposal = converged ? proposal : this.applyCorrection(proposal, validation);

      // 生成STDP更新
      stdpUpdates.push(...this.generateSTDPUpdatesForIteration(proposal, validation, converged));

      onIteration?.(i, proposal, validation);
    }

    return {
      result: currentProposal,
      iterations,
      stdpUpdates,
    };
  }

  /**
   * 生成提案
   */
  private generateProposal(input: string, iteration: number): string {
    // 模拟提案生成
    return `[提案${iteration + 1}] 基于"${input}"的推理结果`;
  }

  /**
   * 验证提案
   */
  private validateProposal(proposal: string, iteration: number): string {
    // 模拟验证过程
    const issues = ['逻辑漏洞', '事实错误', '推理断层'];
    const hasIssue = Math.random() > 0.7;

    if (hasIssue) {
      const issue = issues[Math.floor(Math.random() * issues.length)];
      return `[验证${iteration + 1}] 发现${issue}，需要修正`;
    }

    return `[验证${iteration + 1}] 验证通过，无错误`;
  }

  /**
   * 检查收敛
   */
  private checkConvergence(proposal: string, validation: string): boolean {
    return validation.includes('验证通过');
  }

  /**
   * 应用修正
   */
  private applyCorrection(proposal: string, validation: string): string {
    return `${proposal} [已修正: ${validation}]`;
  }

  /**
   * 生成迭代STDP更新
   */
  private generateSTDPUpdatesForIteration(
    proposal: string,
    validation: string,
    converged: boolean
  ): STDPUpdateRecord[] {
    const now = Date.now();

    return [
      {
        timestamp: now,
        layerType: 'selfEvaluation',
        weightId: `selfplay_${this.iterationCount}`,
        deltaValue: converged ? 0.01 : -0.005,
        updateType: converged ? 'LTP' : 'LTD',
        triggerReason: converged ? '验证通过，路径增强' : '发现错误，路径减弱',
      },
    ];
  }
}

// ==================== 模式3：自双输出+自评判 ====================

/**
 * 自双输出+自评判执行器
 * 双候选生成 + 多维度评判选优
 */
export class SelfEvaluationExecutor {
  private evaluationHistory: EvaluationResult[] = [];
  private maxHistoryLength: number = 100;

  /**
   * 执行自双输出+自评判
   */
  execute(
    input: string,
    cycleCount: number
  ): {
    result: string;
    candidates: CandidateResult[];
    evaluation: EvaluationResult[];
    stdpUpdates: STDPUpdateRecord[];
  } {
    // 每10个刷新周期执行一次
    if (cycleCount % 10 !== 0) {
      return {
        result: '',
        candidates: [],
        evaluation: [],
        stdpUpdates: [],
      };
    }

    // 并行生成2个完整候选
    const candidate1 = this.generateFullCandidate(input, 1);
    const candidate2 = this.generateFullCandidate(input, 2);

    // 切换为评判角色，多维度打分
    const evaluation1 = this.evaluateCandidate(candidate1);
    const evaluation2 = this.evaluateCandidate(candidate2);

    // 选择最优结果
    const bestCandidate =
      evaluation1.totalScore >= evaluation2.totalScore ? candidate1 : candidate2;

    // 更新历史
    this.evaluationHistory.push(evaluation1, evaluation2);
    if (this.evaluationHistory.length > this.maxHistoryLength) {
      this.evaluationHistory = this.evaluationHistory.slice(-this.maxHistoryLength);
    }

    // 生成STDP更新
    const stdpUpdates = this.generateSTDPUpdates(
      [candidate1, candidate2],
      [evaluation1, evaluation2],
      bestCandidate
    );

    return {
      result: bestCandidate.content,
      candidates: [candidate1, candidate2],
      evaluation: [evaluation1, evaluation2],
      stdpUpdates,
    };
  }

  /**
   * 生成完整候选
   */
  private generateFullCandidate(input: string, index: number): CandidateResult {
    return {
      id: generateId(),
      content: `候选${index}: 针对"${input}"的完整回答`,
      score: 0,
      weight: 1,
      stdpWeight: 0.5,
    };
  }

  /**
   * 评判候选
   */
  private evaluateCandidate(candidate: CandidateResult): EvaluationResult {
    // 四个维度打分
    const dimensions: EvaluationDimension[] = [
      {
        name: 'factAccuracy',
        score: Math.floor(Math.random() * 4) + 7, // 7-10分
        weight: EVALUATION_DIMENSION_WEIGHTS.factAccuracy,
      },
      {
        name: 'logicCompleteness',
        score: Math.floor(Math.random() * 4) + 7,
        weight: EVALUATION_DIMENSION_WEIGHTS.logicCompleteness,
      },
      {
        name: 'semanticCoherence',
        score: Math.floor(Math.random() * 4) + 7,
        weight: EVALUATION_DIMENSION_WEIGHTS.semanticCoherence,
      },
      {
        name: 'instructionFollowing',
        score: Math.floor(Math.random() * 4) + 7,
        weight: EVALUATION_DIMENSION_WEIGHTS.instructionFollowing,
      },
    ];

    const totalScore = dimensions.reduce((sum, d) => sum + d.score, 0);

    return {
      candidateId: candidate.id,
      dimensions,
      totalScore,
      feedback: this.generateFeedback(dimensions),
    };
  }

  /**
   * 生成反馈
   */
  private generateFeedback(dimensions: EvaluationDimension[]): string {
    const lowest = dimensions.reduce((min, d) =>
      d.score < min.score ? d : min
    );

    const dimensionNames: Record<string, string> = {
      factAccuracy: '事实准确性',
      logicCompleteness: '逻辑完整性',
      semanticCoherence: '语义连贯性',
      instructionFollowing: '指令遵循度',
    };

    return `整体表现良好，${dimensionNames[lowest.name]}方面可进一步提升`;
  }

  /**
   * 生成STDP更新
   */
  private generateSTDPUpdates(
    candidates: CandidateResult[],
    evaluations: EvaluationResult[],
    bestCandidate: CandidateResult
  ): STDPUpdateRecord[] {
    const now = Date.now();
    const updates: STDPUpdateRecord[] = [];

    for (let i = 0; i < candidates.length; i++) {
      const isBest = candidates[i].id === bestCandidate.id;
      const score = evaluations[i].totalScore;

      updates.push({
        timestamp: now,
        layerType: 'selfEvaluation',
        weightId: candidates[i].id,
        deltaValue: isBest ? 0.01 * (score / 40) : -0.005 * (1 - score / 40),
        updateType: isBest ? 'LTP' : 'LTD',
        triggerReason: `评判得分:${score}/40, ${isBest ? '最优路径' : '次优路径'}`,
      });
    }

    return updates;
  }

  /**
   * 获取评判历史
   */
  getEvaluationHistory(): EvaluationResult[] {
    return [...this.evaluationHistory];
  }
}

// ==================== 闭环优化系统管理器 ====================

/**
 * 闭环优化系统管理器
 * 统一管理三种模式的切换和执行
 */
export class ClosedLoopOptimizationManager {
  private modeSwitcher: ModeSwitcher;
  private selfGenerator: SelfGenerationExecutor;
  private selfPlayer: SelfPlayExecutor;
  private selfEvaluator: SelfEvaluationExecutor;
  private currentMode: OptimizationMode = 'selfGeneration';
  private cycleCount: number = 0;

  constructor() {
    this.modeSwitcher = new ModeSwitcher();
    this.selfGenerator = new SelfGenerationExecutor();
    this.selfPlayer = new SelfPlayExecutor();
    this.selfEvaluator = new SelfEvaluationExecutor();
  }

  /**
   * 执行优化
   */
  execute(
    input: string,
    onModeChange?: (mode: OptimizationMode) => void,
    onIteration?: (iteration: number, proposal: string, validation: string) => void
  ): {
    result: string;
    mode: OptimizationMode;
    details: Record<string, unknown>;
    stdpUpdates: STDPUpdateRecord[];
  } {
    this.cycleCount++;

    // 自动切换模式
    const newMode = this.modeSwitcher.analyzeAndSwitch(input);
    if (newMode !== this.currentMode) {
      this.currentMode = newMode;
      onModeChange?.(newMode);
      console.log(
        `[Optimization] 模式切换: ${this.modeSwitcher.getModeDescription(newMode)}`
      );
    }

    let result: string;
    let details: Record<string, unknown> = {};
    let stdpUpdates: STDPUpdateRecord[] = [];

    switch (this.currentMode) {
      case 'selfGeneration':
        const genResult = this.selfGenerator.execute(input, this.cycleCount);
        result = genResult.result;
        details = { candidates: genResult.candidates };
        stdpUpdates = genResult.stdpUpdates;
        break;

      case 'selfPlay':
        const playResult = this.selfPlayer.execute(input, onIteration);
        result = playResult.result;
        details = { iterations: playResult.iterations };
        stdpUpdates = playResult.stdpUpdates;
        break;

      case 'selfEvaluation':
        const evalResult = this.selfEvaluator.execute(input, this.cycleCount);
        result = evalResult.result || input;
        details = {
          candidates: evalResult.candidates,
          evaluation: evalResult.evaluation,
        };
        stdpUpdates = evalResult.stdpUpdates;
        break;
    }

    return {
      result,
      mode: this.currentMode,
      details,
      stdpUpdates,
    };
  }

  /**
   * 获取当前模式
   */
  getCurrentMode(): OptimizationMode {
    return this.currentMode;
  }

  /**
   * 手动设置模式
   */
  setMode(mode: OptimizationMode): void {
    this.currentMode = mode;
  }

  /**
   * 获取周期计数
   */
  getCycleCount(): number {
    return this.cycleCount;
  }

  /**
   * 获取评判历史
   */
  getEvaluationHistory(): EvaluationResult[] {
    return this.selfEvaluator.getEvaluationHistory();
  }
}

// 导出单例
export const closedLoopOptimization = new ClosedLoopOptimizationManager();
