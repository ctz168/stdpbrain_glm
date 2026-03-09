#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 完整训练优化
Full Training Optimization with STDP
"""

import os
import sys
import json
import time
import random
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from core.brain_architecture import BrainArchitecture, BrainConfig

print("=" * 70)
print("类人脑双系统AI架构 - 完整训练优化")
print("基础模型: Qwen3.5-0.8B-Base")
print("=" * 70)

# ==================== 训练数据 ====================

TRAINING_DATA = [
    # ========== 基础对话 ==========
    {"input": "你好", "output": "你好！我是类人脑AI助手，很高兴为您服务。有什么我可以帮助您的吗？"},
    {"input": "你好啊", "output": "您好！我是基于类人脑双系统架构的AI，具有海马体记忆和STDP学习能力。请问有什么需要帮助的？"},
    {"input": "你是谁", "output": "我是一个基于类人脑双系统架构的人工智能，拥有100Hz高刷新推理能力和STDP学习机制。"},
    {"input": "请问你是谁？", "output": "我是类人脑AI助手，基于Qwen2.5-0.5B模型，具有海马体记忆系统和自闭环优化能力。"},
    {"input": "介绍一下你自己", "output": "我是类人脑双系统AI架构的智能助手。我的核心特性包括：100Hz高刷新推理、STDP学习机制、海马体记忆系统、三模式自闭环优化。"},
    {"input": "谢谢", "output": "不客气！如果还有其他问题，随时可以问我。我很乐意为您提供帮助。"},
    {"input": "谢谢你的帮助", "output": "很高兴能帮助到您！如有其他问题，请随时提问。"},
    {"input": "再见", "output": "再见！祝您生活愉快，期待下次与您交流。"},
    {"input": "晚安", "output": "晚安！祝您有个好梦，明天见！"},
    {"input": "早上好", "output": "早上好！新的一天开始了，有什么我可以帮助您的吗？"},
    
    # ========== 知识问答 ==========
    {"input": "什么是人工智能？", "output": "人工智能（AI）是模拟人类智能的技术，包括学习、推理、感知、理解语言等能力。它使机器能够执行通常需要人类智能才能完成的任务。"},
    {"input": "什么是深度学习？", "output": "深度学习是机器学习的一个分支，使用多层神经网络进行特征学习和模式识别。它能够自动从数据中学习层次化的特征表示。"},
    {"input": "什么是神经网络？", "output": "神经网络是模拟生物神经系统的计算模型，由大量互联的节点（神经元）组成，能够学习和识别复杂的模式。"},
    {"input": "什么是STDP？", "output": "STDP（脉冲时序依赖可塑性）是一种基于时序的突触可塑性学习规则。当突触前神经元先于突触后神经元激活时，突触增强（LTP）；反之则减弱（LTD）。"},
    {"input": "什么是类脑计算？", "output": "类脑计算是模拟人脑结构和功能的计算方式，具有低功耗、高效率、强适应性的特点。它采用事件驱动和并行处理的方式处理信息。"},
    {"input": "什么是海马体？", "output": "海马体是大脑中负责记忆编码和空间导航的关键区域。在AI中，海马体模块负责情景记忆的存储和回忆，包括EC、DG、CA3、CA1等子区域。"},
    {"input": "什么是注意力机制？", "output": "注意力机制让模型能够聚焦于输入的重要部分，提高处理效率。它模拟人类视觉注意力的选择性，动态分配计算资源。"},
    {"input": "什么是Transformer？", "output": "Transformer是一种基于自注意力机制的神经网络架构，广泛应用于自然语言处理。它能够并行处理序列数据，捕捉长距离依赖关系。"},
    {"input": "100Hz刷新周期是什么意思？", "output": "100Hz刷新周期意味着每秒执行100次刷新，即每10毫秒完成一次完整的推理周期。这与人脑的gamma脑波频率相近。"},
    {"input": "权重双轨拆分是什么？", "output": "权重双轨拆分是将模型权重分为两部分：90%静态权重（永久冻结，保持预训练能力）和10%动态权重（可更新，支持在线学习）。"},
    
    # ========== 数学计算 ==========
    {"input": "1+1等于几？", "output": "1+1=2"},
    {"input": "2+3等于几？", "output": "2+3=5"},
    {"input": "10+5等于几？", "output": "10+5=15"},
    {"input": "100+200等于几？", "output": "100+200=300"},
    {"input": "10-5等于几？", "output": "10-5=5"},
    {"input": "100-30等于几？", "output": "100-30=70"},
    {"input": "3乘以4等于几？", "output": "3×4=12"},
    {"input": "5乘以6等于几？", "output": "5×6=30"},
    {"input": "20除以4等于几？", "output": "20÷4=5"},
    {"input": "100除以5等于几？", "output": "100÷5=20"},
    {"input": "2的3次方等于几？", "output": "2³=8"},
    {"input": "10的平方等于几？", "output": "10²=100"},
    
    # ========== 逻辑推理 ==========
    {"input": "如果A>B，B>C，那么A和C谁大？", "output": "根据传递性，A>B且B>C，所以A>C。因此A比C大。"},
    {"input": "所有鸟都会飞吗？", "output": "不是所有鸟都会飞。企鹅、鸵鸟、几维鸟等是不会飞的鸟类的典型例子。它们在进化过程中失去了飞行能力。"},
    {"input": "太阳从哪个方向升起？", "output": "太阳从东方升起，从西方落下。这是由于地球自西向东自转造成的。"},
    {"input": "水在什么温度结冰？", "output": "在标准大气压下，水在0摄氏度（32华氏度）时结冰。"},
    {"input": "一年有多少天？", "output": "平年有365天，闰年有366天。闰年每4年出现一次，2月份有29天。"},
    {"input": "地球绕太阳转一圈需要多久？", "output": "地球绕太阳公转一圈大约需要365.25天，即一年。"},
    
    # ========== 专业问答 ==========
    {"input": "什么是自闭环优化？", "output": "自闭环优化是三种模式的自动切换：自生成组合输出、自博弈竞争优化、自双输出+自评判。系统根据任务自动选择最优模式。"},
    {"input": "海马体记忆系统如何工作？", "output": "海马体记忆系统包含：内嗅皮层EC（特征编码）、齿状回DG（模式分离）、CA3区（情景记忆存储）、CA1区（时序编码）和尖波涟漪SWR（离线回放巩固）。"},
    {"input": "什么是LTP和LTD？", "output": "LTP（长时程增强）是突触强度增加的过程，LTD（长时程减弱）是突触强度减弱的过程。它们是STDP学习的核心机制。"},
    
    # ========== 编程相关 ==========
    {"input": "什么是Python？", "output": "Python是一种高级编程语言，以简洁易读的语法著称。它广泛应用于Web开发、数据科学、人工智能、自动化脚本等领域。"},
    {"input": "什么是机器学习？", "output": "机器学习是人工智能的一个分支，让计算机能够从数据中学习规律，而无需显式编程。主要类型包括监督学习、无监督学习和强化学习。"},
    {"input": "什么是GPU？", "output": "GPU（图形处理器）最初用于图形渲染，现在广泛用于深度学习计算。它具有大量并行计算单元，适合处理矩阵运算。"},
    
    # ========== 生活常识 ==========
    {"input": "如何保持健康？", "output": "保持健康的方法包括：均衡饮食、规律运动、充足睡眠、保持良好心态、定期体检、戒烟限酒。"},
    {"input": "多喝水有什么好处？", "output": "多喝水有助于：促进新陈代谢、帮助消化、保持皮肤健康、调节体温、排出体内毒素、预防肾结石。"},
    {"input": "如何提高学习效率？", "output": "提高学习效率的方法：制定学习计划、使用番茄工作法、主动回忆和间隔重复、保持专注、适当休息、多种学习方式结合。"},
    {"input": "什么是健康饮食？", "output": "健康饮食包括：多吃蔬菜水果、选择全谷物、适量摄入蛋白质、减少糖和盐的摄入、保持饮食多样化。"},
    
    # ========== 情感交流 ==========
    {"input": "我今天心情不好", "output": "很抱歉听到这个。每个人都会有情绪低落的时候，这是正常的。您可以试着做些喜欢的事情，或者和朋友聊聊天。需要我陪您聊聊吗？"},
    {"input": "我感到很累", "output": "您辛苦了。记得适当休息，保证睡眠。如果长期感到疲惫，建议检查一下身体状况。有什么我可以帮您的吗？"},
    {"input": "我很开心", "output": "太好了！很高兴听到您心情愉快。保持积极的心态对身心健康都很有益。有什么开心的事情想分享吗？"},
    {"input": "我有点焦虑", "output": "焦虑是常见的情绪反应。您可以尝试深呼吸、冥想或轻度运动来缓解。如果焦虑持续，建议寻求专业帮助。"},
]

print(f"\n训练数据: {len(TRAINING_DATA)} 条")

# ==================== 初始化架构 ====================

print("\n" + "=" * 70)
print("[1/4] 初始化架构...")
print("=" * 70)

config = BrainConfig(
    model_path=str(Path(__file__).parent.parent / "models" / "qwen3.5-0.8b"),
    max_new_tokens=128,
    temperature=0.7
)

brain = BrainArchitecture(config)
result = brain.initialize()

if not result["success"]:
    print(f"初始化失败: {result['message']}")
    sys.exit(1)

print(f"✓ 架构初始化成功")
print(f"  模型: {result['model_info']['name']}")
print(f"  参数: {result['model_info']['total_params']/1e6:.1f}M")

# ==================== 执行训练 ====================

print("\n" + "=" * 70)
print("[2/4] 执行STDP训练...")
print("=" * 70)

# 训练配置
EPOCHS = 5
LEARNING_RATE = 5e-6

# 数据增强：打乱并重复
random.seed(42)
augmented_data = TRAINING_DATA * 2
random.shuffle(augmented_data)

print(f"增强后数据: {len(augmented_data)} 条")
print(f"训练轮次: {EPOCHS}")
print(f"学习率: {LEARNING_RATE}")

training_log = []
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    epoch_loss = 0
    epoch_stdp = 0
    
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"{'='*50}")
    
    # 每轮随机选择部分数据
    samples = random.sample(augmented_data, min(20, len(augmented_data)))
    
    for i, sample in enumerate(samples):
        # 执行训练步骤
        loss, info = brain.engine.train_step(
            sample["input"],
            sample["output"],
            torch.optim.AdamW(
                filter(lambda p: p.requires_grad, brain.model.parameters()),
                lr=LEARNING_RATE
            )
        )
        
        epoch_loss += loss
        epoch_stdp += info.get("stdp_updates", 0)
        
        # 打印进度
        if (i + 1) % 5 == 0:
            print(f"  样本 {i+1}/{len(samples)} | Loss: {loss:.4f}")
    
    avg_loss = epoch_loss / len(samples)
    epoch_time = time.time() - epoch_start
    
    training_log.append({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "stdp_updates": epoch_stdp,
        "time": epoch_time
    })
    
    print(f"\n  Epoch {epoch + 1} 完成:")
    print(f"    平均损失: {avg_loss:.4f}")
    print(f"    STDP更新: {epoch_stdp}")
    print(f"    耗时: {epoch_time:.1f}s")

total_time = time.time() - start_time

print(f"\n{'='*70}")
print(f"训练完成!")
print(f"{'='*70}")
print(f"总耗时: {total_time:.1f}秒")

# ==================== 评估 ====================

print("\n" + "=" * 70)
print("[3/4] 评估训练效果...")
print("=" * 70)

brain.engine.clear_history()

test_cases = [
    "你好",
    "什么是STDP？",
    "1+1等于几？",
    "如果A>B，B>C，那么A和C谁大？",
    "如何保持健康？"
]

print("\n推理测试:")
for prompt in test_cases:
    output, metadata = brain.engine.infer(prompt)
    print(f"\n输入: {prompt}")
    print(f"输出: {output[:150]}...")
    print(f"耗时: {metadata['cycle_time_ms']:.1f}ms")

# ==================== 保存 ====================

print("\n" + "=" * 70)
print("[4/4] 保存训练结果...")
print("=" * 70)

# 保存训练日志
log_path = Path("./models/qwen3.5-0.8b/full_training_log.json")
with open(log_path, 'w') as f:
    json.dump({
        "config": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "data_size": len(augmented_data)
        },
        "training_log": training_log,
        "total_time": total_time,
        "stdp_stats": brain.engine.stdp.get_statistics(),
        "hippocampus_stats": brain.engine.hippocampus.get_statistics()
    }, f, indent=2)

print(f"✓ 训练日志已保存: {log_path}")

# 保存模型
try:
    save_path = Path("./models/trained")
    save_path.mkdir(parents=True, exist_ok=True)
    
    brain.model.save_pretrained(str(save_path))
    brain.tokenizer.save_pretrained(str(save_path))
    
    print(f"✓ 模型已保存: {save_path}")
except Exception as e:
    print(f"模型保存跳过: {e}")

# ==================== 总结 ====================

print("\n" + "=" * 70)
print("训练总结")
print("=" * 70)
print(f"训练轮次: {EPOCHS}")
print(f"训练样本: {len(augmented_data)}")
print(f"总耗时: {total_time:.1f}秒")
print(f"初始损失: {training_log[0]['avg_loss']:.4f}")
print(f"最终损失: {training_log[-1]['avg_loss']:.4f}")
print(f"损失下降: {training_log[0]['avg_loss'] - training_log[-1]['avg_loss']:.4f}")
print(f"总STDP更新: {sum(log['stdp_updates'] for log in training_log)}")
print(f"海马体记忆: {brain.engine.hippocampus.get_statistics()['total_memories']} 条")
print("=" * 70)
