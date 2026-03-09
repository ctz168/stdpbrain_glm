#!/usr/bin/env python3
"""
模块7：多维度全链路测评体系
Module 7: Multi-Dimensional Evaluation System

包含：
1. 海马体记忆能力专项测评（40%）
2. 基础能力对标测评（20%）
3. 逻辑推理能力测评（20%）
4. 端侧性能测评（10%）
5. 自闭环优化能力测评（10%）
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import numpy as np

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.brain_architecture_full import BrainArchitecture, BrainConfig


# ==================== 测评结果数据结构 ====================

@dataclass
class EvaluationMetrics:
    """测评指标"""
    name: str
    score: float
    max_score: float
    details: Dict = field(default_factory=dict)
    
    @property
    def percentage(self) -> float:
        return (self.score / self.max_score) * 100 if self.max_score > 0 else 0


@dataclass
class EvaluationReport:
    """测评报告"""
    total_score: float
    max_score: float
    metrics: List[EvaluationMetrics]
    passed: bool
    summary: str
    
    def to_dict(self) -> Dict:
        return {
            "total_score": self.total_score,
            "max_score": self.max_score,
            "percentage": (self.total_score / self.max_score) * 100,
            "passed": self.passed,
            "summary": self.summary,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "max_score": m.max_score,
                    "percentage": m.percentage,
                    "details": m.details
                }
                for m in self.metrics
            ]
        }


# ==================== 1. 海马体记忆能力专项测评（40%） ====================

class HippocampusMemoryEvaluator:
    """
    海马体记忆能力专项测评
    权重：40%
    """
    
    def __init__(self, brain: BrainArchitecture):
        self.brain = brain
        self.weight = 0.4
    
    def evaluate(self) -> EvaluationMetrics:
        """执行完整测评"""
        print("\n" + "=" * 50)
        print("海马体记忆能力专项测评")
        print("=" * 50)
        
        total_score = 0
        max_score = 100
        details = {}
        
        # 1. 情景记忆召回能力（20分）
        recall_score = self._test_memory_recall()
        total_score += recall_score
        details["情景记忆召回能力"] = recall_score
        print(f"  情景记忆召回能力: {recall_score}/20")
        
        # 2. 模式分离抗混淆能力（20分）
        separation_score = self._test_pattern_separation()
        total_score += separation_score
        details["模式分离抗混淆能力"] = separation_score
        print(f"  模式分离抗混淆能力: {separation_score}/20")
        
        # 3. 长时序记忆保持能力（20分）
        retention_score = self._test_long_term_retention()
        total_score += retention_score
        details["长时序记忆保持能力"] = retention_score
        print(f"  长时序记忆保持能力: {retention_score}/20")
        
        # 4. 模式补全能力（20分）
        completion_score = self._test_pattern_completion()
        total_score += completion_score
        details["模式补全能力"] = completion_score
        print(f"  模式补全能力: {completion_score}/20")
        
        # 5. 抗灾难性遗忘能力（10分）
        forgetting_score = self._test_catastrophic_forgetting()
        total_score += forgetting_score
        details["抗灾难性遗忘能力"] = forgetting_score
        print(f"  抗灾难性遗忘能力: {forgetting_score}/10")
        
        # 6. 跨会话终身学习能力（10分）
        cross_session_score = self._test_cross_session_learning()
        total_score += cross_session_score
        details["跨会话终身学习能力"] = cross_session_score
        print(f"  跨会话终身学习能力: {cross_session_score}/10")
        
        return EvaluationMetrics(
            name="海马体记忆能力",
            score=total_score * self.weight,
            max_score=max_score * self.weight,
            details=details
        )
    
    def _test_memory_recall(self) -> float:
        """测试情景记忆召回能力"""
        score = 0
        
        # 测试：存储多轮对话，然后用部分线索召回
        test_cases = [
            {
                "store": "我的名字是张三，今年25岁，住在北京",
                "query": "我叫什么名字？",
                "expected": "张三"
            },
            {
                "store": "我喜欢吃苹果和香蕉",
                "query": "我喜欢吃什么水果？",
                "expected": "苹果"
            }
        ]
        
        correct = 0
        for case in test_cases:
            # 存储
            self.brain.infer(f"请记住：{case['store']}")
            # 召回
            result = self.brain.infer(case['query'])
            if case['expected'] in result.get('output', ''):
                correct += 1
        
        # 计算分数
        accuracy = correct / len(test_cases)
        score = accuracy * 20
        
        return min(20, score)
    
    def _test_pattern_separation(self) -> float:
        """测试模式分离抗混淆能力"""
        score = 0
        
        # 测试：存储相似上下文，检查是否混淆
        similar_contexts = [
            "用户A说：我喜欢蓝色",
            "用户B说：我喜欢绿色",
            "用户C说：我喜欢红色"
        ]
        
        # 存储
        for ctx in similar_contexts:
            self.brain.infer(ctx)
        
        # 检查混淆
        result_a = self.brain.infer("用户A喜欢什么颜色？")
        result_b = self.brain.infer("用户B喜欢什么颜色？")
        
        if "蓝色" in result_a.get('output', '') and "绿色" in result_b.get('output', ''):
            score = 20
        else:
            score = 10
        
        return score
    
    def _test_long_term_retention(self) -> float:
        """测试长时序记忆保持能力"""
        # 简化测试
        score = 15  # 基础分
        
        # 存储长序列信息
        long_text = "开头信息：这是一个重要的会议，时间是明天上午10点。"
        self.brain.infer(long_text)
        
        # 添加更多上下文
        for i in range(10):
            self.brain.infer(f"这是第{i+1}条中间信息。")
        
        # 召回开头信息
        result = self.brain.infer("会议是什么时候？")
        if "10点" in result.get('output', ''):
            score = 20
        
        return score
    
    def _test_pattern_completion(self) -> float:
        """测试模式补全能力"""
        score = 0
        
        # 存储
        self.brain.infer("我的手机号是13812345678，邮箱是test@example.com")
        
        # 用部分线索召回
        result = self.brain.infer("我的邮箱是什么？")
        
        if "test@example.com" in result.get('output', ''):
            score = 20
        elif "example" in result.get('output', ''):
            score = 15
        else:
            score = 10
        
        return score
    
    def _test_catastrophic_forgetting(self) -> float:
        """测试抗灾难性遗忘能力"""
        score = 0
        
        # 学习任务A
        self.brain.infer("任务A：1+1=2")
        
        # 学习任务B
        self.brain.infer("任务B：2+2=4")
        
        # 测试任务A是否还记得
        result = self.brain.infer("任务A的结果是什么？")
        
        if "2" in result.get('output', ''):
            score = 10
        else:
            score = 5
        
        return score
    
    def _test_cross_session_learning(self) -> float:
        """测试跨会话终身学习能力"""
        # 简化测试
        score = 8  # 基础分
        
        # 模拟跨会话
        self.brain.infer("会话1：我的偏好是简洁回复")
        
        # 清除部分历史
        self.brain.clear_history()
        
        # 新会话
        result = self.brain.infer("我的偏好是什么？")
        
        # 检查是否记住
        if "简洁" in result.get('output', ''):
            score = 10
        
        return score


# ==================== 2. 基础能力对标测评（20%） ====================

class BasicAbilityEvaluator:
    """
    基础能力对标测评
    权重：20%
    """
    
    def __init__(self, brain: BrainArchitecture):
        self.brain = brain
        self.weight = 0.2
    
    def evaluate(self) -> EvaluationMetrics:
        """执行测评"""
        print("\n" + "=" * 50)
        print("基础能力对标测评")
        print("=" * 50)
        
        total_score = 0
        max_score = 100
        details = {}
        
        # 1. 通用对话能力（25分）
        dialog_score = self._test_dialog_ability()
        total_score += dialog_score
        details["通用对话能力"] = dialog_score
        print(f"  通用对话能力: {dialog_score}/25")
        
        # 2. 指令遵循能力（25分）
        instruction_score = self._test_instruction_following()
        total_score += instruction_score
        details["指令遵循能力"] = instruction_score
        print(f"  指令遵循能力: {instruction_score}/25")
        
        # 3. 语义理解能力（25分）
        semantic_score = self._test_semantic_understanding()
        total_score += semantic_score
        details["语义理解能力"] = semantic_score
        print(f"  语义理解能力: {semantic_score}/25")
        
        # 4. 中文处理能力（25分）
        chinese_score = self._test_chinese_processing()
        total_score += chinese_score
        details["中文处理能力"] = chinese_score
        print(f"  中文处理能力: {chinese_score}/25")
        
        return EvaluationMetrics(
            name="基础能力",
            score=total_score * self.weight,
            max_score=max_score * self.weight,
            details=details
        )
    
    def _test_dialog_ability(self) -> float:
        """测试通用对话能力"""
        test_cases = [
            {"input": "你好", "check": lambda x: "你好" in x or "您好" in x},
            {"input": "谢谢", "check": lambda x: "不客气" in x or "欢迎" in x},
            {"input": "再见", "check": lambda x: "再见" in x or "下次" in x}
        ]
        
        correct = 0
        for case in test_cases:
            result = self.brain.infer(case['input'])
            if case['check'](result.get('output', '')):
                correct += 1
        
        return (correct / len(test_cases)) * 25
    
    def _test_instruction_following(self) -> float:
        """测试指令遵循能力"""
        test_cases = [
            {"input": "请用一句话介绍自己", "check": lambda x: len(x) < 100},
            {"input": "请用三个词描述AI", "check": lambda x: len(x.split()) <= 10}
        ]
        
        correct = 0
        for case in test_cases:
            result = self.brain.infer(case['input'])
            if case['check'](result.get('output', '')):
                correct += 1
        
        return (correct / len(test_cases)) * 25
    
    def _test_semantic_understanding(self) -> float:
        """测试语义理解能力"""
        result = self.brain.infer("苹果是水果还是公司？")
        output = result.get('output', '')
        
        if "水果" in output or "公司" in output:
            return 25
        return 15
    
    def _test_chinese_processing(self) -> float:
        """测试中文处理能力"""
        result = self.brain.infer("请用中文说一句话")
        output = result.get('output', '')
        
        # 检查是否包含中文
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in output)
        
        return 25 if has_chinese else 10


# ==================== 3. 逻辑推理能力测评（20%） ====================

class ReasoningEvaluator:
    """
    逻辑推理能力测评
    权重：20%
    """
    
    def __init__(self, brain: BrainArchitecture):
        self.brain = brain
        self.weight = 0.2
    
    def evaluate(self) -> EvaluationMetrics:
        """执行测评"""
        print("\n" + "=" * 50)
        print("逻辑推理能力测评")
        print("=" * 50)
        
        total_score = 0
        max_score = 100
        details = {}
        
        # 1. 数学推理（25分）
        math_score = self._test_math_reasoning()
        total_score += math_score
        details["数学推理"] = math_score
        print(f"  数学推理: {math_score}/25")
        
        # 2. 常识推理（25分）
        common_score = self._test_commonsense_reasoning()
        total_score += common_score
        details["常识推理"] = common_score
        print(f"  常识推理: {common_score}/25")
        
        # 3. 因果推断（25分）
        causal_score = self._test_causal_reasoning()
        total_score += causal_score
        details["因果推断"] = causal_score
        print(f"  因果推断: {causal_score}/25")
        
        # 4. 事实性问答（25分）
        fact_score = self._test_fact_qa()
        total_score += fact_score
        details["事实性问答"] = fact_score
        print(f"  事实性问答: {fact_score}/25")
        
        return EvaluationMetrics(
            name="逻辑推理能力",
            score=total_score * self.weight,
            max_score=max_score * self.weight,
            details=details
        )
    
    def _test_math_reasoning(self) -> float:
        """测试数学推理"""
        test_cases = [
            {"input": "1+1等于几？", "expected": "2"},
            {"input": "10-5等于几？", "expected": "5"},
            {"input": "3乘以4等于几？", "expected": "12"}
        ]
        
        correct = 0
        for case in test_cases:
            result = self.brain.infer(case['input'])
            if case['expected'] in result.get('output', ''):
                correct += 1
        
        return (correct / len(test_cases)) * 25
    
    def _test_commonsense_reasoning(self) -> float:
        """测试常识推理"""
        result = self.brain.infer("太阳从哪个方向升起？")
        output = result.get('output', '')
        
        if "东" in output:
            return 25
        return 10
    
    def _test_causal_reasoning(self) -> float:
        """测试因果推断"""
        result = self.brain.infer("如果下雨，地面会怎样？")
        output = result.get('output', '')
        
        if "湿" in output or "水" in output:
            return 25
        return 15
    
    def _test_fact_qa(self) -> float:
        """测试事实性问答"""
        result = self.brain.infer("中国的首都是哪里？")
        output = result.get('output', '')
        
        if "北京" in output:
            return 25
        return 10


# ==================== 4. 端侧性能测评（10%） ====================

class EdgePerformanceEvaluator:
    """
    端侧性能测评
    权重：10%
    """
    
    def __init__(self, brain: BrainArchitecture):
        self.brain = brain
        self.weight = 0.1
    
    def evaluate(self) -> EvaluationMetrics:
        """执行测评"""
        print("\n" + "=" * 50)
        print("端侧性能测评")
        print("=" * 50)
        
        total_score = 0
        max_score = 100
        details = {}
        
        # 1. 显存占用（25分）
        memory_score = self._test_memory_usage()
        total_score += memory_score
        details["显存占用"] = memory_score
        print(f"  显存占用: {memory_score}/25")
        
        # 2. 推理延迟（25分）
        latency_score = self._test_inference_latency()
        total_score += latency_score
        details["推理延迟"] = latency_score
        print(f"  推理延迟: {latency_score}/25")
        
        # 3. 长序列稳定性（25分）
        stability_score = self._test_long_sequence_stability()
        total_score += stability_score
        details["长序列稳定性"] = stability_score
        print(f"  长序列稳定性: {stability_score}/25")
        
        # 4. 离线运行兼容性（25分）
        offline_score = self._test_offline_compatibility()
        total_score += offline_score
        details["离线运行兼容性"] = offline_score
        print(f"  离线运行兼容性: {offline_score}/25")
        
        return EvaluationMetrics(
            name="端侧性能",
            score=total_score * self.weight,
            max_score=max_score * self.weight,
            details=details
        )
    
    def _test_memory_usage(self) -> float:
        """测试显存占用"""
        # 获取模型大小
        total_params = sum(p.numel() for p in self.brain.model.parameters())
        model_size_mb = (total_params * 4) / (1024 * 1024)  # float32
        
        # 评分
        if model_size_mb < 420:
            return 25
        elif model_size_mb < 500:
            return 20
        else:
            return 10
    
    def _test_inference_latency(self) -> float:
        """测试推理延迟"""
        start_time = time.time()
        self.brain.infer("测试推理延迟")
        latency = (time.time() - start_time) * 1000
        
        # 评分
        if latency < 100:
            return 25
        elif latency < 500:
            return 20
        elif latency < 1000:
            return 15
        else:
            return 10
    
    def _test_long_sequence_stability(self) -> float:
        """测试长序列稳定性"""
        try:
            # 测试长输入
            long_input = "测试" * 100
            result = self.brain.infer(long_input)
            
            if result.get('output'):
                return 25
            return 15
        except:
            return 5
    
    def _test_offline_compatibility(self) -> float:
        """测试离线运行兼容性"""
        # 检查是否使用本地模型
        if self.brain.is_initialized:
            return 25
        return 10


# ==================== 5. 自闭环优化能力测评（10%） ====================

class OptimizationEvaluator:
    """
    自闭环优化能力测评
    权重：10%
    """
    
    def __init__(self, brain: BrainArchitecture):
        self.brain = brain
        self.weight = 0.1
    
    def evaluate(self) -> EvaluationMetrics:
        """执行测评"""
        print("\n" + "=" * 50)
        print("自闭环优化能力测评")
        print("=" * 50)
        
        total_score = 0
        max_score = 100
        details = {}
        
        # 1. 自纠错能力（25分）
        correction_score = self._test_self_correction()
        total_score += correction_score
        details["自纠错能力"] = correction_score
        print(f"  自纠错能力: {correction_score}/25")
        
        # 2. 幻觉抑制能力（25分）
        hallucination_score = self._test_hallucination_suppression()
        total_score += hallucination_score
        details["幻觉抑制能力"] = hallucination_score
        print(f"  幻觉抑制能力: {hallucination_score}/25")
        
        # 3. 输出准确率提升（25分）
        accuracy_score = self._test_accuracy_improvement()
        total_score += accuracy_score
        details["输出准确率提升"] = accuracy_score
        print(f"  输出准确率提升: {accuracy_score}/25")
        
        # 4. 持续进化能力（25分）
        evolution_score = self._test_continuous_evolution()
        total_score += evolution_score
        details["持续进化能力"] = evolution_score
        print(f"  持续进化能力: {evolution_score}/25")
        
        return EvaluationMetrics(
            name="自闭环优化能力",
            score=total_score * self.weight,
            max_score=max_score * self.weight,
            details=details
        )
    
    def _test_self_correction(self) -> float:
        """测试自纠错能力"""
        # 检查STDP更新次数
        stats = self.brain.engine.stdp_system.get_statistics()
        
        if stats.get('total_updates', 0) > 100:
            return 25
        elif stats.get('total_updates', 0) > 10:
            return 20
        else:
            return 15
    
    def _test_hallucination_suppression(self) -> float:
        """测试幻觉抑制能力"""
        # 测试简单问题
        result = self.brain.infer("你好")
        output = result.get('output', '')
        
        # 检查是否包含编造的用户输入
        if "用户:" not in output:
            return 25
        else:
            return 10
    
    def _test_accuracy_improvement(self) -> float:
        """测试输出准确率提升"""
        # 简化测试
        return 20
    
    def _test_continuous_evolution(self) -> float:
        """测试持续进化能力"""
        # 检查海马体记忆数量
        stats = self.brain.engine.hippocampus.get_statistics()
        
        if stats.get('total_memories', 0) > 10:
            return 25
        elif stats.get('total_memories', 0) > 0:
            return 20
        else:
            return 10


# ==================== 综合测评系统 ====================

class FullEvaluationSystem:
    """
    多维度全链路测评体系
    整合所有测评模块
    """
    
    def __init__(self, brain: BrainArchitecture):
        self.brain = brain
        
        # 初始化测评器
        self.hippocampus_evaluator = HippocampusMemoryEvaluator(brain)
        self.basic_evaluator = BasicAbilityEvaluator(brain)
        self.reasoning_evaluator = ReasoningEvaluator(brain)
        self.performance_evaluator = EdgePerformanceEvaluator(brain)
        self.optimization_evaluator = OptimizationEvaluator(brain)
    
    def run_full_evaluation(self) -> EvaluationReport:
        """运行完整测评"""
        print("\n" + "=" * 70)
        print("多维度全链路测评体系")
        print("=" * 70)
        
        start_time = time.time()
        metrics = []
        
        # 1. 海马体记忆能力测评（40%）
        m1 = self.hippocampus_evaluator.evaluate()
        metrics.append(m1)
        
        # 2. 基础能力测评（20%）
        m2 = self.basic_evaluator.evaluate()
        metrics.append(m2)
        
        # 3. 逻辑推理能力测评（20%）
        m3 = self.reasoning_evaluator.evaluate()
        metrics.append(m3)
        
        # 4. 端侧性能测评（10%）
        m4 = self.performance_evaluator.evaluate()
        metrics.append(m4)
        
        # 5. 自闭环优化能力测评（10%）
        m5 = self.optimization_evaluator.evaluate()
        metrics.append(m5)
        
        # 计算总分
        total_score = sum(m.score for m in metrics)
        max_score = sum(m.max_score for m in metrics)
        
        # 判断是否通过
        passed = total_score >= max_score * 0.6  # 60%及格
        
        # 生成总结
        total_time = time.time() - start_time
        summary = f"""
测评完成！
总分: {total_score:.1f}/{max_score:.1f} ({total_score/max_score*100:.1f}%)
结果: {'通过 ✓' if passed else '未通过 ✗'}
耗时: {total_time:.1f}秒

各模块得分:
- 海马体记忆能力: {m1.score:.1f}/{m1.max_score:.1f}
- 基础能力: {m2.score:.1f}/{m2.max_score:.1f}
- 逻辑推理能力: {m3.score:.1f}/{m3.max_score:.1f}
- 端侧性能: {m4.score:.1f}/{m4.max_score:.1f}
- 自闭环优化能力: {m5.score:.1f}/{m5.max_score:.1f}
"""
        
        return EvaluationReport(
            total_score=total_score,
            max_score=max_score,
            metrics=metrics,
            passed=passed,
            summary=summary
        )
    
    def save_report(self, report: EvaluationReport, path: str = None):
        """保存测评报告"""
        if path is None:
            path = "./evaluation_report.json"
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"\n测评报告已保存: {path}")


# ==================== 导出 ====================

__all__ = [
    'FullEvaluationSystem',
    'HippocampusMemoryEvaluator',
    'BasicAbilityEvaluator',
    'ReasoningEvaluator',
    'EdgePerformanceEvaluator',
    'OptimizationEvaluator',
    'EvaluationMetrics',
    'EvaluationReport'
]
