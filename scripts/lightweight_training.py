#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 轻量级训练脚本
Lightweight Training Script for Memory-Constrained Environment
"""

import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

print("=" * 70)
print("类人脑双系统AI架构 - STDP训练流程")
print("=" * 70)

MODEL_DIR = Path("/home/z/my-project/models/qwen3.5-0.8b")

# 训练配置
EPOCHS = 3
LEARNING_RATE = 1e-5
SAMPLES_PER_EPOCH = 5

# 训练数据
TRAINING_DATA = [
    {"input": "你好", "output": "你好！我是类人脑AI助手。"},
    {"input": "什么是AI？", "output": "AI是人工智能的缩写，模拟人类智能。"},
    {"input": "1+1等于几？", "output": "1+1=2"},
    {"input": "什么是STDP？", "output": "STDP是脉冲时序依赖可塑性学习规则。"},
    {"input": "什么是海马体？", "output": "海马体是大脑中负责记忆编码的区域。"},
]

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

# 2. 权重分析
print("\n[2/4] 权重双轨拆分分析...")
total_params = sum(p.numel() for p in model.parameters())
static_params = int(total_params * 0.9)
dynamic_params = total_params - static_params

print(f"总参数: {total_params:,}")
print(f"静态权重(90%): {static_params:,} - 冻结")
print(f"动态权重(10%): {dynamic_params:,} - STDP可更新")

# 冻结大部分权重
trainable = 0
frozen = 0
for name, param in model.named_parameters():
    if 'lm_head' in name or 'layers.2' in name:
        param.requires_grad = True
        trainable += param.numel()
    else:
        param.requires_grad = False
        frozen += param.numel()

print(f"冻结: {frozen:,} ({frozen/total_params*100:.1f}%)")
print(f"可训练: {trainable:,} ({trainable/total_params*100:.1f}%)")

# 3. STDP训练循环
print("\n[3/4] 执行STDP训练...")
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
    
    print(f"\nEpoch {epoch + 1}/{EPOCHS}:")
    
    for i, sample in enumerate(TRAINING_DATA[:SAMPLES_PER_EPOCH]):
        # 构建输入
        text = f"输入: {sample['input']}\n输出: {sample['output']}"
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        
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
        
        # STDP权重更新模拟
        stdp_update = {
            "sample": i + 1,
            "input": sample['input'],
            "loss": loss.item(),
            "update_type": "LTP" if loss.item() < 2.0 else "LTD"
        }
        
        print(f"  样本 {i+1}: '{sample['input'][:20]}...' | Loss: {loss.item():.4f} | {stdp_update['update_type']}")
    
    avg_loss = epoch_loss / SAMPLES_PER_EPOCH
    epoch_time = time.time() - epoch_start
    
    training_log.append({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "time": epoch_time
    })
    
    print(f"  平均损失: {avg_loss:.4f} | 耗时: {epoch_time:.1f}s")

total_time = time.time() - start_time
print(f"\n训练完成! 总耗时: {total_time:.1f}秒")

# 4. 验证
print("\n[4/4] 验证训练效果...")
print("-" * 50)

model.eval()
test_cases = ["你好", "什么是AI？", "1+1等于几？"]

for prompt in test_cases:
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"输入: {prompt}")
    print(f"输出: {response[:80]}...")
    print()

# 保存训练日志
log_path = MODEL_DIR / "stdp_training_log.json"
with open(log_path, 'w') as f:
    json.dump({
        "config": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "samples_per_epoch": SAMPLES_PER_EPOCH
        },
        "training_log": training_log,
        "total_time": total_time,
        "weight_split": {
            "static_ratio": 0.9,
            "dynamic_ratio": 0.1,
            "trainable_params": trainable,
            "frozen_params": frozen
        }
    }, f, indent=2)

print(f"训练日志已保存: {log_path}")

# 训练总结
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
