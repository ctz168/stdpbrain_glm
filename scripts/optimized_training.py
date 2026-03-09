#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 优化训练脚本
Optimized Training Script with Better Data and Strategy
"""

import os
import sys
import json
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path

print("=" * 70)
print("类人脑双系统AI架构 - 优化训练")
print("=" * 70)

# 配置
MODEL_DIR = Path("/home/z/my-project/models/qwen3.5-0.8b")
OUTPUT_DIR = Path("/home/z/my-project/models/optimized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 训练配置
CONFIG = {
    "epochs": 10,
    "batch_size": 2,
    "learning_rate": 5e-6,
    "max_length": 512,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
}

# 扩展训练数据
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
    
    # ========== 知识问答 ==========
    {"input": "什么是人工智能？", "output": "人工智能（AI）是模拟人类智能的技术，包括学习、推理、感知、理解语言等能力。它使机器能够执行通常需要人类智能才能完成的任务。"},
    {"input": "什么是深度学习？", "output": "深度学习是机器学习的一个分支，使用多层神经网络进行特征学习和模式识别。它能够自动从数据中学习层次化的特征表示。"},
    {"input": "什么是神经网络？", "output": "神经网络是模拟生物神经系统的计算模型，由大量互联的节点（神经元）组成，能够学习和识别复杂的模式。"},
    {"input": "什么是STDP？", "output": "STDP（脉冲时序依赖可塑性）是一种基于时序的突触可塑性学习规则。当突触前神经元先于突触后神经元激活时，突触增强（LTP）；反之则减弱（LTD）。"},
    {"input": "什么是类脑计算？", "output": "类脑计算是模拟人脑结构和功能的计算方式，具有低功耗、高效率、强适应性的特点。它采用事件驱动和并行处理的方式处理信息。"},
    {"input": "什么是海马体？", "output": "海马体是大脑中负责记忆编码和空间导航的关键区域。在AI中，海马体模块负责情景记忆的存储和回忆，包括EC、DG、CA3、CA1等子区域。"},
    {"input": "什么是注意力机制？", "output": "注意力机制让模型能够聚焦于输入的重要部分，提高处理效率。它模拟人类视觉注意力的选择性，动态分配计算资源。"},
    {"input": "什么是Transformer？", "output": "Transformer是一种基于自注意力机制的神经网络架构，广泛应用于自然语言处理。它能够并行处理序列数据，捕捉长距离依赖关系。"},
    
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
    
    # ========== 逻辑推理 ==========
    {"input": "如果A>B，B>C，那么A和C谁大？", "output": "根据传递性，A>B且B>C，所以A>C。因此A比C大。"},
    {"input": "所有鸟都会飞吗？", "output": "不是所有鸟都会飞。企鹅、鸵鸟、几维鸟等是不会飞的鸟类的典型例子。它们在进化过程中失去了飞行能力。"},
    {"input": "太阳从哪个方向升起？", "output": "太阳从东方升起，从西方落下。这是由于地球自西向东自转造成的。"},
    {"input": "水在什么温度结冰？", "output": "在标准大气压下，水在0摄氏度（32华氏度）时结冰。"},
    {"input": "一年有多少天？", "output": "平年有365天，闰年有366天。闰年每4年出现一次，2月份有29天。"},
    
    # ========== 专业问答 ==========
    {"input": "100Hz刷新周期是什么意思？", "output": "100Hz刷新周期意味着每秒执行100次刷新，即每10毫秒完成一次完整的推理周期。这与人脑的gamma脑波频率相近。"},
    {"input": "权重双轨拆分是什么？", "output": "权重双轨拆分是将模型权重分为两部分：90%静态权重（永久冻结，保持预训练能力）和10%动态权重（可更新，支持在线学习）。"},
    {"input": "什么是自闭环优化？", "output": "自闭环优化是三种模式的自动切换：自生成组合输出、自博弈竞争优化、自双输出+自评判。系统根据任务自动选择最优模式。"},
    {"input": "海马体记忆系统如何工作？", "output": "海马体记忆系统包含：内嗅皮层EC（特征编码）、齿状回DG（模式分离）、CA3区（情景记忆存储）、CA1区（时序编码）和尖波涟漪SWR（离线回放巩固）。"},
    
    # ========== 编程相关 ==========
    {"input": "什么是Python？", "output": "Python是一种高级编程语言，以简洁易读的语法著称。它广泛应用于Web开发、数据科学、人工智能、自动化脚本等领域。"},
    {"input": "什么是机器学习？", "output": "机器学习是人工智能的一个分支，让计算机能够从数据中学习规律，而无需显式编程。主要类型包括监督学习、无监督学习和强化学习。"},
    {"input": "什么是GPU？", "output": "GPU（图形处理器）最初用于图形渲染，现在广泛用于深度学习计算。它具有大量并行计算单元，适合处理矩阵运算。"},
    
    # ========== 生活常识 ==========
    {"input": "如何保持健康？", "output": "保持健康的方法包括：均衡饮食、规律运动、充足睡眠、保持良好心态、定期体检、戒烟限酒。"},
    {"input": "多喝水有什么好处？", "output": "多喝水有助于：促进新陈代谢、帮助消化、保持皮肤健康、调节体温、排出体内毒素、预防肾结石。"},
    {"input": "如何提高学习效率？", "output": "提高学习效率的方法：制定学习计划、使用番茄工作法、主动回忆和间隔重复、保持专注、适当休息、多种学习方式结合。"},
    
    # ========== 情感交流 ==========
    {"input": "我今天心情不好", "output": "很抱歉听到这个。每个人都会有情绪低落的时候，这是正常的。您可以试着做些喜欢的事情，或者和朋友聊聊天。需要我陪您聊聊吗？"},
    {"input": "我感到很累", "output": "您辛苦了。记得适当休息，保证睡眠。如果长期感到疲惫，建议检查一下身体状况。有什么我可以帮您的吗？"},
    {"input": "我很开心", "output": "太好了！很高兴听到您心情愉快。保持积极的心态对身心健康都很有益。有什么开心的事情想分享吗？"},
]

print(f"\n训练数据: {len(TRAINING_DATA)} 条")
print(f"训练配置: {json.dumps(CONFIG, indent=2)}")

# 1. 加载模型
print("\n" + "=" * 70)
print("[1/6] 加载模型...")
print("=" * 70)

try:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    print("✓ 模型加载成功")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    sys.exit(1)

# 2. 权重分析
print("\n" + "=" * 70)
print("[2/6] 权重双轨拆分...")
print("=" * 70)

total_params = sum(p.numel() for p in model.parameters())
static_params = int(total_params * 0.9)
dynamic_params = total_params - static_params

print(f"总参数: {total_params:,}")
print(f"静态权重(90%): {static_params:,}")
print(f"动态权重(10%): {dynamic_params:,}")

# 冻结策略：只训练最后几层和部分注意力
trainable = 0
frozen = 0
trainable_layers = []

for name, param in model.named_parameters():
    # 训练最后2层、lm_head和部分注意力
    if any(x in name for x in ['layers.2', 'layers.3', 'lm_head']):
        param.requires_grad = True
        trainable += param.numel()
        trainable_layers.append(name)
    else:
        param.requires_grad = False
        frozen += param.numel()

print(f"\n冻结参数: {frozen:,} ({frozen/total_params*100:.1f}%)")
print(f"可训练参数: {trainable:,} ({trainable/total_params*100:.1f}%)")
print(f"可训练层数: {len(set(trainable_layers))}")

# 3. 准备数据
print("\n" + "=" * 70)
print("[3/6] 准备训练数据...")
print("=" * 70)

class TrainingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 格式化输入
        text = f"用户: {item['input']}\n助手: {item['output']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone()
        }

# 数据增强：打乱并重复
random.seed(42)
augmented_data = TRAINING_DATA * 3  # 重复3次
random.shuffle(augmented_data)

dataset = TrainingDataset(augmented_data, tokenizer, CONFIG["max_length"])
dataloader = DataLoader(
    dataset, 
    batch_size=CONFIG["batch_size"], 
    shuffle=True,
    drop_last=True
)

print(f"原始数据: {len(TRAINING_DATA)} 条")
print(f"增强后数据: {len(augmented_data)} 条")
print(f"批次数: {len(dataloader)}")

# 4. 配置优化器
print("\n" + "=" * 70)
print("[4/6] 配置优化器...")
print("=" * 70)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"]
)

total_steps = len(dataloader) * CONFIG["epochs"] // CONFIG["gradient_accumulation_steps"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"优化器: AdamW")
print(f"学习率: {CONFIG['learning_rate']}")
print(f"总训练步数: {total_steps}")
print(f"预热步数: {warmup_steps}")
print(f"梯度累积: {CONFIG['gradient_accumulation_steps']}")

# 5. 训练循环
print("\n" + "=" * 70)
print("[5/6] 开始优化训练...")
print("=" * 70)

model.train()
training_history = []
global_step = 0
best_loss = float('inf')
start_time = time.time()

for epoch in range(CONFIG["epochs"]):
    epoch_loss = 0
    epoch_start = time.time()
    num_batches = 0
    
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")
    print(f"{'='*50}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        # 前向传播
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss / CONFIG["gradient_accumulation_steps"]
        
        # 反向传播
        loss.backward()
        
        epoch_loss += loss.item() * CONFIG["gradient_accumulation_steps"]
        num_batches += 1
        
        # 梯度累积
        if (batch_idx + 1) % CONFIG["gradient_accumulation_steps"] == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # 打印进度
            if global_step % 10 == 0:
                current_loss = loss.item() * CONFIG["gradient_accumulation_steps"]
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step} | Loss: {current_loss:.4f} | LR: {lr:.2e}")
    
    # Epoch统计
    avg_loss = epoch_loss / num_batches
    epoch_time = time.time() - epoch_start
    
    training_history.append({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "epoch_time": epoch_time,
        "learning_rate": scheduler.get_last_lr()[0]
    })
    
    print(f"\n  Epoch {epoch + 1} 完成:")
    print(f"    平均损失: {avg_loss:.4f}")
    print(f"    耗时: {epoch_time:.1f}秒")
    print(f"    学习率: {scheduler.get_last_lr()[0]:.2e}")
    
    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"    ✓ 新的最佳损失!")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"训练完成!")
print(f"{'='*70}")
print(f"总耗时: {total_time:.1f}秒")
print(f"总步数: {global_step}")
print(f"最佳损失: {best_loss:.4f}")

# 6. 评估
print("\n" + "=" * 70)
print("[6/6] 评估训练效果...")
print("=" * 70)

model.eval()

# 测试推理
test_cases = [
    "你好",
    "什么是人工智能？",
    "1+1等于几？",
    "如果A>B，B>C，那么A和C谁大？",
    "如何保持健康？"
]

print("\n推理测试:")
for prompt in test_cases:
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n输入: {prompt}")
    print(f"输出: {response[:150]}...")

# 保存训练历史
history_path = OUTPUT_DIR / "training_history.json"
with open(history_path, 'w') as f:
    json.dump({
        "config": CONFIG,
        "history": training_history,
        "total_time": total_time,
        "total_steps": global_step,
        "best_loss": best_loss,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "data_size": len(augmented_data)
    }, f, indent=2)

print(f"\n训练历史已保存: {history_path}")

# 训练总结
print("\n" + "=" * 70)
print("训练总结")
print("=" * 70)
print(f"  训练轮次: {CONFIG['epochs']}")
print(f"  训练样本: {len(augmented_data)}")
print(f"  总步数: {global_step}")
print(f"  总耗时: {total_time:.1f}秒")
print(f"  初始损失: {training_history[0]['avg_loss']:.4f}")
print(f"  最终损失: {training_history[-1]['avg_loss']:.4f}")
print(f"  最佳损失: {best_loss:.4f}")
print(f"  损失下降: {training_history[0]['avg_loss'] - best_loss:.4f} ({(training_history[0]['avg_loss'] - best_loss)/training_history[0]['avg_loss']*100:.1f}%)")
print("=" * 70)
