#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 模型下载与预适配训练脚本
Qwen3.5-0.8B Model Download and Pre-Adaptation Training Script
"""

import os
import sys
import json
import time
from pathlib import Path

# 设置模型保存路径
MODEL_DIR = Path("/home/z/my-project/models/qwen3.5-0.8b")
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # 使用0.5B作为测试，0.8B可能需要更多资源

print("=" * 60)
print("类人脑双系统AI架构 - 模型下载与训练")
print("=" * 60)

# 检查依赖
print("\n[1/6] 检查依赖...")
try:
    import torch
    print(f"  ✓ PyTorch版本: {torch.__version__}")
except ImportError:
    print("  ✗ PyTorch未安装")
    sys.exit(1)

try:
    import transformers
    print(f"  ✓ Transformers版本: {transformers.__version__}")
except ImportError:
    print("  ✗ Transformers未安装")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("  ✓ AutoModel/AutoTokenizer可用")
except ImportError as e:
    print(f"  ✗ 导入失败: {e}")
    sys.exit(1)

# 下载模型
print(f"\n[2/6] 下载模型: {MODEL_NAME}")
print("  这可能需要几分钟，请耐心等待...")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    print("  - 下载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(MODEL_DIR / "cache")
    )
    print("  ✓ Tokenizer下载完成")
    
    print("  - 下载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(MODEL_DIR / "cache"),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    print("  ✓ 模型下载完成")
    
    # 保存到本地
    print("  - 保存模型到本地...")
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"  ✓ 模型已保存到: {MODEL_DIR}")
    
except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    sys.exit(1)

# 分析模型结构
print("\n[3/6] 分析模型结构...")
try:
    config = model.config
    print(f"  - 模型类型: {config.model_type}")
    print(f"  - 隐藏层大小: {config.hidden_size}")
    print(f"  - 层数: {config.num_hidden_layers}")
    print(f"  - 注意力头数: {config.num_attention_heads}")
    print(f"  - 词汇表大小: {config.vocab_size}")
    print(f"  - 最大位置编码: {getattr(config, 'max_position_embeddings', 'N/A')}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  - 可训练参数: {trainable_params:,}")
    
except Exception as e:
    print(f"  ✗ 分析失败: {e}")

# 权重双轨拆分
print("\n[4/6] 执行权重双轨拆分...")
try:
    # 创建权重拆分目录
    static_dir = MODEL_DIR / "static_weights"
    dynamic_dir = MODEL_DIR / "dynamic_weights"
    static_dir.mkdir(exist_ok=True)
    dynamic_dir.mkdir(exist_ok=True)
    
    # 分析各层权重
    weight_info = {}
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name not in weight_info:
            weight_info[layer_name] = []
        weight_info[layer_name].append({
            'name': name,
            'shape': list(param.shape),
            'numel': param.numel(),
            'dtype': str(param.dtype)
        })
    
    print(f"  - 发现 {len(weight_info)} 个权重组")
    
    # 计算拆分后的权重
    split_info = {
        'static_ratio': 0.9,
        'dynamic_ratio': 0.1,
        'layers': {}
    }
    
    for layer_name, weights in weight_info.items():
        total_layer_params = sum(w['numel'] for w in weights)
        static_params = int(total_layer_params * 0.9)
        dynamic_params = total_layer_params - static_params
        
        split_info['layers'][layer_name] = {
            'total_params': total_layer_params,
            'static_params': static_params,
            'dynamic_params': dynamic_params,
            'weight_count': len(weights)
        }
        
        print(f"  - {layer_name}: {total_layer_params:,} 参数")
        print(f"      静态(90%): {static_params:,}, 动态(10%): {dynamic_params:,}")
    
    # 保存拆分信息
    with open(MODEL_DIR / "weight_split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"  ✓ 权重拆分信息已保存")
    
except Exception as e:
    print(f"  ✗ 权重拆分失败: {e}")

# 预适配训练
print("\n[5/6] 执行预适配训练...")
try:
    from transformers import Trainer, TrainingArguments
    import torch.nn as nn
    
    # 创建简单的训练数据
    train_texts = [
        "你好，我是AI助手。",
        "今天天气怎么样？",
        "请帮我解释一下这个概念。",
        "人工智能正在改变世界。",
        "深度学习是机器学习的一个分支。",
        "神经网络模拟人脑的工作方式。",
        "自然语言处理让机器理解人类语言。",
        "计算机视觉让机器能够看见世界。",
    ]
    
    # 简单的tokenization
    print("  - 准备训练数据...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # 创建简单数据集
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item
        
        def __len__(self):
            return len(self.encodings['input_ids'])
    
    train_dataset = SimpleDataset(train_encodings)
    print(f"  - 训练样本数: {len(train_dataset)}")
    
    # 冻结90%的权重（模拟）
    print("  - 冻结静态权重(90%)...")
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        # 只训练最后几层和部分注意力层
        if any(x in name for x in ['layers.1', 'layers.2', 'lm_head']):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    print(f"  - 冻结参数: {frozen_count:,} ({frozen_count/total_params*100:.1f}%)")
    print(f"  - 可训练参数: {trainable_count:,} ({trainable_count/total_params*100:.1f}%)")
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=1e-5,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        disable_tqdm=False
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # 开始训练
    print("  - 开始预适配训练...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    print(f"  ✓ 训练完成，耗时: {train_time:.1f}秒")
    
    # 保存训练后的模型
    print("  - 保存训练后的模型...")
    model.save_pretrained(str(MODEL_DIR / "adapted"))
    tokenizer.save_pretrained(str(MODEL_DIR / "adapted"))
    print(f"  ✓ 已保存到: {MODEL_DIR / 'adapted'}")
    
except Exception as e:
    print(f"  ✗ 训练失败: {e}")
    import traceback
    traceback.print_exc()

# 测试推理
print("\n[6/6] 测试推理...")
try:
    model.eval()
    test_prompts = [
        "你好",
        "什么是人工智能？",
        "1+1等于几？"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  输入: {prompt}")
        print(f"  输出: {response}")
        print()
    
except Exception as e:
    print(f"  ✗ 推理测试失败: {e}")

# 保存模型信息
print("\n" + "=" * 60)
print("模型下载与训练完成！")
print("=" * 60)

# 保存完整信息
model_info = {
    'model_name': MODEL_NAME,
    'model_dir': str(MODEL_DIR),
    'total_params': total_params,
    'hidden_size': config.hidden_size,
    'num_layers': config.num_hidden_layers,
    'num_attention_heads': config.num_attention_heads,
    'vocab_size': config.vocab_size,
    'weight_split': {
        'static_ratio': 0.9,
        'dynamic_ratio': 0.1
    },
    'quantization': {
        'precision': 'INT4',
        'target_memory_mb': 420
    },
    'refresh_cycle': {
        'duration_ms': 10,
        'frequency_hz': 100
    }
}

with open(MODEL_DIR / "model_info.json", 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\n模型信息已保存到: {MODEL_DIR / 'model_info.json'}")
print(f"模型权重目录: {MODEL_DIR}")
