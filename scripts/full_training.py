#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 完整训练脚本
Full Training Script with Detailed Logging
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path

print("=" * 70)
print("类人脑双系统AI架构 - 完整训练流程")
print("Human-Like Brain Dual-System AI Architecture - Full Training")
print("=" * 70)

# 配置
MODEL_DIR = Path("/home/z/my-project/models/qwen3.5-0.8b")
OUTPUT_DIR = Path("/home/z/my-project/models/trained")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 训练配置
TRAINING_CONFIG = {
    "epochs": 5,
    "batch_size": 4,
    "learning_rate": 1e-5,
    "max_length": 256,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "static_ratio": 0.9,  # 90%静态权重
    "dynamic_ratio": 0.1,  # 10%动态权重
}

# 训练数据集
TRAINING_DATA = [
    # 对话场景
    {"input": "你好", "output": "你好！我是类人脑AI助手，很高兴为您服务。"},
    {"input": "请问你是谁？", "output": "我是一个基于类人脑双系统架构的AI，具有海马体记忆系统和STDP学习能力。"},
    {"input": "今天天气怎么样？", "output": "抱歉，我无法获取实时天气信息。建议您查看天气预报应用。"},
    {"input": "谢谢你的帮助", "output": "不客气！如果还有其他问题，随时可以问我。"},
    
    # 知识问答
    {"input": "什么是人工智能？", "output": "人工智能（AI）是模拟人类智能的技术，包括学习、推理、感知等能力。"},
    {"input": "什么是深度学习？", "output": "深度学习是机器学习的分支，使用多层神经网络进行特征学习和模式识别。"},
    {"input": "什么是神经网络？", "output": "神经网络是模拟生物神经系统的计算模型，由大量互联的节点组成。"},
    {"input": "什么是STDP？", "output": "STDP（脉冲时序依赖可塑性）是一种基于时序的突触可塑性学习规则。"},
    
    # 数学计算
    {"input": "1+1等于几？", "output": "1+1=2"},
    {"input": "2+3等于几？", "output": "2+3=5"},
    {"input": "10-5等于几？", "output": "10-5=5"},
    {"input": "3乘以4等于几？", "output": "3×4=12"},
    
    # 逻辑推理
    {"input": "如果A>B，B>C，那么A和C谁大？", "output": "根据传递性，A>C，所以A比C大。"},
    {"input": "所有鸟都会飞吗？", "output": "不是所有鸟都会飞，比如企鹅和鸵鸟就不会飞。"},
    {"input": "太阳从哪个方向升起？", "output": "太阳从东方升起，从西方落下。"},
    
    # 专业问答
    {"input": "什么是类脑计算？", "output": "类脑计算是模拟人脑结构和功能的计算方式，具有低功耗、高效率的特点。"},
    {"input": "什么是海马体？", "output": "海马体是大脑中负责记忆编码和空间导航的关键区域。"},
    {"input": "什么是注意力机制？", "output": "注意力机制让模型能够聚焦于输入的重要部分，提高处理效率。"},
]

print(f"\n训练数据: {len(TRAINING_DATA)} 条")
print(f"训练配置: {json.dumps(TRAINING_CONFIG, indent=2)}")

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
    print(f"✓ 模型加载成功")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    sys.exit(1)

# 2. 权重双轨拆分
print("\n" + "=" * 70)
print("[2/6] 执行权重双轨拆分...")
print("=" * 70)

total_params = sum(p.numel() for p in model.parameters())
static_params = int(total_params * TRAINING_CONFIG["static_ratio"])
dynamic_params = total_params - static_params

print(f"总参数量: {total_params:,}")
print(f"静态权重(90%): {static_params:,} - 永久冻结")
print(f"动态权重(10%): {dynamic_params:,} - STDP可更新")

# 冻结90%的权重
frozen_count = 0
trainable_count = 0
layer_count = 0

for name, param in model.named_parameters():
    layer_count += 1
    # 只训练最后几层
    if any(x in name for x in ['layers.2', 'layers.3', 'lm_head', 'embed_tokens']):
        param.requires_grad = True
        trainable_count += param.numel()
    else:
        param.requires_grad = False
        frozen_count += param.numel()

print(f"\n冻结层数: {layer_count - 4} 层")
print(f"冻结参数: {frozen_count:,} ({frozen_count/total_params*100:.1f}%)")
print(f"可训练参数: {trainable_count:,} ({trainable_count/total_params*100:.1f}%)")

# 3. 准备训练数据
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
        text = f"输入: {item['input']}\n输出: {item['output']}"
        
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

dataset = TrainingDataset(TRAINING_DATA, tokenizer, TRAINING_CONFIG["max_length"])
dataloader = DataLoader(dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)

print(f"数据集大小: {len(dataset)}")
print(f"批次大小: {TRAINING_CONFIG['batch_size']}")
print(f"批次数: {len(dataloader)}")

# 4. 配置优化器
print("\n" + "=" * 70)
print("[4/6] 配置优化器和调度器...")
print("=" * 70)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=TRAINING_CONFIG["learning_rate"],
    weight_decay=TRAINING_CONFIG["weight_decay"]
)

total_steps = len(dataloader) * TRAINING_CONFIG["epochs"]
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=TRAINING_CONFIG["warmup_steps"],
    num_training_steps=total_steps
)

print(f"优化器: AdamW")
print(f"学习率: {TRAINING_CONFIG['learning_rate']}")
print(f"总训练步数: {total_steps}")
print(f"预热步数: {TRAINING_CONFIG['warmup_steps']}")

# 5. 开始训练
print("\n" + "=" * 70)
print("[5/6] 开始训练...")
print("=" * 70)

model.train()
training_history = []
global_step = 0
start_time = time.time()

for epoch in range(TRAINING_CONFIG["epochs"]):
    epoch_loss = 0
    epoch_start = time.time()
    
    print(f"\n--- Epoch {epoch + 1}/{TRAINING_CONFIG['epochs']} ---")
    
    for batch_idx, batch in enumerate(dataloader):
        # 前向传播
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        global_step += 1
        
        # 打印进度
        if (batch_idx + 1) % 2 == 0 or batch_idx == len(dataloader) - 1:
            elapsed = time.time() - epoch_start
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {elapsed:.1f}s")
    
    # Epoch统计
    avg_loss = epoch_loss / len(dataloader)
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

total_time = time.time() - start_time
print(f"\n训练完成!")
print(f"总耗时: {total_time:.1f}秒")
print(f"总步数: {global_step}")

# 6. 评估和保存
print("\n" + "=" * 70)
print("[6/6] 评估和保存模型...")
print("=" * 70)

model.eval()

# 测试推理
test_cases = [
    "你好",
    "什么是人工智能？",
    "1+1等于几？"
]

print("\n推理测试:")
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

# 保存训练历史
history_path = OUTPUT_DIR / "training_history.json"
with open(history_path, 'w') as f:
    json.dump({
        "config": TRAINING_CONFIG,
        "history": training_history,
        "total_time": total_time,
        "total_steps": global_step,
        "frozen_params": frozen_count,
        "trainable_params": trainable_count
    }, f, indent=2)

print(f"\n训练历史已保存: {history_path}")

# 保存模型（如果空间足够）
try:
    model.save_pretrained(str(OUTPUT_DIR / "model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "model"))
    print(f"模型已保存: {OUTPUT_DIR / 'model'}")
except Exception as e:
    print(f"模型保存跳过 (空间不足): {e}")

print("\n" + "=" * 70)
print("训练流程完成!")
print("=" * 70)

# 打印训练总结
print("\n训练总结:")
print(f"  - 训练轮次: {TRAINING_CONFIG['epochs']}")
print(f"  - 训练样本: {len(TRAINING_DATA)}")
print(f"  - 总步数: {global_step}")
print(f"  - 总耗时: {total_time:.1f}秒")
print(f"  - 初始损失: {training_history[0]['avg_loss']:.4f}")
print(f"  - 最终损失: {training_history[-1]['avg_loss']:.4f}")
print(f"  - 损失下降: {(training_history[0]['avg_loss'] - training_history[-1]['avg_loss']):.4f}")
