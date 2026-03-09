#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 轻量优化训练
Lightweight Optimized Training
"""

import os
import sys
import json
import time
import random
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

print("=" * 70)
print("类人脑双系统AI架构 - 轻量优化训练")
print("=" * 70)

MODEL_DIR = Path("/home/z/my-project/models/qwen3.5-0.8b")

# 训练配置
EPOCHS = 5
SAMPLES_PER_EPOCH = 10
LEARNING_RATE = 5e-6

# 扩展训练数据
TRAINING_DATA = [
    # 基础对话
    {"input": "你好", "output": "你好！我是类人脑AI助手，很高兴为您服务。"},
    {"input": "你是谁", "output": "我是基于类人脑双系统架构的AI，具有海马体记忆和STDP学习能力。"},
    {"input": "谢谢", "output": "不客气！如果还有其他问题，随时可以问我。"},
    
    # 知识问答
    {"input": "什么是人工智能？", "output": "人工智能（AI）是模拟人类智能的技术，包括学习、推理、感知等能力。"},
    {"input": "什么是深度学习？", "output": "深度学习是机器学习的分支，使用多层神经网络进行特征学习和模式识别。"},
    {"input": "什么是STDP？", "output": "STDP是脉冲时序依赖可塑性，一种基于时序的突触可塑性学习规则。"},
    {"input": "什么是海马体？", "output": "海马体是大脑中负责记忆编码的关键区域，包括EC、DG、CA3、CA1等子区域。"},
    {"input": "100Hz刷新周期是什么？", "output": "100Hz刷新周期意味着每秒执行100次刷新，即每10毫秒完成一次推理周期。"},
    {"input": "权重双轨拆分是什么？", "output": "权重双轨拆分将模型权重分为90%静态权重（冻结）和10%动态权重（可更新）。"},
    
    # 数学计算
    {"input": "1+1等于几？", "output": "1+1=2"},
    {"input": "2+3等于几？", "output": "2+3=5"},
    {"input": "10-5等于几？", "output": "10-5=5"},
    {"input": "3乘以4等于几？", "output": "3×4=12"},
    
    # 逻辑推理
    {"input": "如果A>B，B>C，那么A和C谁大？", "output": "根据传递性，A>B且B>C，所以A>C，A比C大。"},
    {"input": "太阳从哪个方向升起？", "output": "太阳从东方升起，从西方落下。"},
    
    # 专业问答
    {"input": "什么是类脑计算？", "output": "类脑计算是模拟人脑结构和功能的计算方式，具有低功耗、高效率的特点。"},
    {"input": "什么是注意力机制？", "output": "注意力机制让模型能够聚焦于输入的重要部分，提高处理效率。"},
    {"input": "什么是Transformer？", "output": "Transformer是基于自注意力机制的神经网络架构，广泛应用于自然语言处理。"},
    
    # 编程相关
    {"input": "什么是Python？", "output": "Python是一种高级编程语言，以简洁易读的语法著称，广泛应用于AI和数据科学。"},
    {"input": "什么是机器学习？", "output": "机器学习让计算机能够从数据中学习规律，无需显式编程。"},
    
    # 生活常识
    {"input": "如何保持健康？", "output": "保持健康需要均衡饮食、规律运动、充足睡眠和良好心态。"},
    {"input": "多喝水有什么好处？", "output": "多喝水有助于促进新陈代谢、帮助消化、保持皮肤健康。"},
]

print(f"\n训练数据: {len(TRAINING_DATA)} 条")

# 1. 加载模型
print("\n[1/4] 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
print("✓ 模型加载成功")

# 2. 权重配置
print("\n[2/4] 配置权重...")
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数: {total_params:,}")

# 冻结大部分权重
trainable = 0
for name, param in model.named_parameters():
    if 'lm_head' in name or 'layers.3' in name:
        param.requires_grad = True
        trainable += param.numel()
    else:
        param.requires_grad = False

print(f"可训练参数: {trainable:,} ({trainable/total_params*100:.1f}%)")

# 3. 训练
print("\n[3/4] 开始训练...")
print("-" * 50)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

training_log = []
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_loss = 0
    epoch_start = time.time()
    
    # 随机选择样本
    samples = random.sample(TRAINING_DATA, min(SAMPLES_PER_EPOCH, len(TRAINING_DATA)))
    
    print(f"\nEpoch {epoch + 1}/{EPOCHS}:")
    
    for i, sample in enumerate(samples):
        # 构建输入
        text = f"用户: {sample['input']}\n助手: {sample['output']}"
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        
        # 前向传播
        model.train()
        outputs = model(
            input_ids=inputs['input_ids'],
            labels=inputs['input_ids']
        )
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # 清理内存
        del outputs, loss
        gc.collect()
        
        # 打印进度
        print(f"  样本 {i+1}/{len(samples)}: '{sample['input'][:15]}...' | Loss: {epoch_loss/(i+1):.4f}")
    
    avg_loss = epoch_loss / len(samples)
    epoch_time = time.time() - epoch_start
    
    training_log.append({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "time": epoch_time
    })
    
    print(f"  平均损失: {avg_loss:.4f} | 耗时: {epoch_time:.1f}s")

total_time = time.time() - start_time
print(f"\n训练完成! 总耗时: {total_time:.1f}秒")

# 4. 评估
print("\n[4/4] 评估效果...")
print("-" * 50)

model.eval()
test_cases = ["你好", "什么是人工智能？", "1+1等于几？", "什么是STDP？"]

for prompt in test_cases:
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n输入: {prompt}")
    print(f"输出: {response[:100]}...")

# 保存日志
log_path = MODEL_DIR / "optimized_training_log.json"
with open(log_path, 'w') as f:
    json.dump({
        "epochs": EPOCHS,
        "samples_per_epoch": SAMPLES_PER_EPOCH,
        "learning_rate": LEARNING_RATE,
        "training_log": training_log,
        "total_time": total_time,
        "trainable_params": trainable
    }, f, indent=2)

print(f"\n训练日志已保存: {log_path}")

# 总结
print("\n" + "=" * 70)
print("训练总结")
print("=" * 70)
print(f"训练轮次: {EPOCHS}")
print(f"每轮样本: {SAMPLES_PER_EPOCH}")
print(f"初始损失: {training_log[0]['avg_loss']:.4f}")
print(f"最终损失: {training_log[-1]['avg_loss']:.4f}")
print(f"损失下降: {training_log[0]['avg_loss'] - training_log[-1]['avg_loss']:.4f}")
print(f"总耗时: {total_time:.1f}秒")
print("=" * 70)
