#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 完整测试
Full Architecture Test
"""

import os
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.brain_architecture_full import BrainArchitecture, BrainConfig
from core.training_module import TrainingManager
from core.evaluation_module import FullEvaluationSystem

print("=" * 70)
print("类人脑双系统全闭环AI架构 - 完整测试")
print("=" * 70)

# 配置
config = BrainConfig(
    model_path="./models/qwen3.5-0.8b",
    refresh_cycle_ms=10.0,
    refresh_rate_hz=100.0,
    narrow_window_size=2,
    static_ratio=0.9,
    dynamic_ratio=0.1
)

# 初始化架构
print("\n[1/4] 初始化架构...")
brain = BrainArchitecture(config)
result = brain.initialize()

if not result['success']:
    print(f"初始化失败: {result['message']}")
    sys.exit(1)

print(f"\n✓ 架构初始化成功")
print(f"  模型: {result['model_info']['name']}")
print(f"  参数: {result['model_info']['total_params']/1e6:.1f}M")
print(f"  隐藏层: {result['model_info']['hidden_size']}")
print(f"  层数: {result['model_info']['num_layers']}")

# 测试推理
print("\n[2/4] 测试推理引擎...")
test_cases = [
    "你好",
    "1+1等于几？",
    "什么是STDP？"
]

for user_input in test_cases:
    print(f"\n输入: {user_input}")
    result = brain.infer(user_input)
    print(f"输出: {result['output'][:100]}...")
    if 'metadata' in result:
        print(f"模式: {result['metadata'].get('optimization_mode', 'N/A')}")
        print(f"周期: {result['metadata'].get('cycle_id', 0)}")

# 测试训练
print("\n[3/4] 测试训练模块...")
training_manager = TrainingManager(brain, config)

# 简单训练测试
training_data = [
    {"input": "测试问题1", "output": "测试回答1"},
    {"input": "测试问题2", "output": "测试回答2"}
]

train_result = brain.train(training_data, epochs=1)
print(f"训练完成: 损失 {train_result.get('final_loss', 0):.4f}")

# 测试测评
print("\n[4/4] 测试测评系统...")
evaluation_system = FullEvaluationSystem(brain)

# 运行简化测评
print("\n运行基础能力测评...")
from core.evaluation_module import BasicAbilityEvaluator
basic_eval = BasicAbilityEvaluator(brain)
basic_result = basic_eval.evaluate()
print(f"基础能力得分: {basic_result.score:.1f}/{basic_result.max_score:.1f}")

# 获取状态
print("\n" + "=" * 70)
print("架构状态")
print("=" * 70)
status = brain.get_status()
print(f"初始化: {status.get('initialized', False)}")
print(f"运行中: {status.get('is_running', False)}")
print(f"周期数: {status.get('cycle_count', 0)}")
print(f"当前模式: {status.get('current_mode', 'N/A')}")

stdp_stats = status.get('stdp_stats', {})
print(f"\nSTDP统计:")
print(f"  总更新: {stdp_stats.get('total_updates', 0)}")
print(f"  LTP: {stdp_stats.get('ltp_count', 0)}")
print(f"  LTD: {stdp_stats.get('ltd_count', 0)}")

hippo_stats = status.get('hippocampus_stats', {})
print(f"\n海马体统计:")
print(f"  总记忆: {hippo_stats.get('total_memories', 0)}")
print(f"  平均强度: {hippo_stats.get('avg_strength', 0):.2f}")

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
