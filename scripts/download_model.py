#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 模型下载脚本
Model Download Script
"""

import os
import sys
import json
import time
from pathlib import Path

# 配置
MODEL_DIR = Path(__file__).parent.parent / "models" / "qwen3.5-0.8b"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("=" * 60)
print("类人脑双系统AI架构 - 模型下载")
print("=" * 60)

def check_dependencies():
    """检查依赖"""
    print("\n[1/4] 检查依赖...")
    
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch未安装")
        print("  请运行: pip install torch")
        sys.exit(1)
    
    try:
        import transformers
        print(f"  ✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("  ✗ Transformers未安装")
        print("  请运行: pip install transformers")
        sys.exit(1)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("  ✓ AutoModel可用")
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        sys.exit(1)

def download_model():
    """下载模型"""
    print(f"\n[2/4] 下载模型: {MODEL_NAME}")
    print("  这可能需要几分钟，请耐心等待...")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 下载Tokenizer
    print("  - 下载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(MODEL_DIR / "cache")
    )
    print("  ✓ Tokenizer下载完成")
    
    # 下载模型
    print("  - 下载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(MODEL_DIR / "cache"),
        torch_dtype="auto",
        low_cpu_mem_usage=True
    )
    print("  ✓ 模型下载完成")
    
    # 保存到本地
    print("  - 保存模型到本地...")
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"  ✓ 模型已保存到: {MODEL_DIR}")
    
    return model, tokenizer

def analyze_model(model):
    """分析模型"""
    print("\n[3/4] 分析模型结构...")
    
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  - 模型类型: {config.model_type}")
    print(f"  - 隐藏层大小: {config.hidden_size}")
    print(f"  - 层数: {config.num_hidden_layers}")
    print(f"  - 注意力头数: {config.num_attention_heads}")
    print(f"  - 词汇表大小: {config.vocab_size}")
    print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    
    return {
        "model_name": MODEL_NAME,
        "model_dir": str(MODEL_DIR),
        "total_params": total_params,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "weight_split": {
            "static_ratio": 0.9,
            "dynamic_ratio": 0.1
        }
    }

def save_info(model_info):
    """保存模型信息"""
    print("\n[4/4] 保存模型信息...")
    
    # 保存模型信息
    info_path = MODEL_DIR / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"  ✓ 模型信息已保存: {info_path}")
    
    # 保存权重拆分信息
    split_info = {
        "static_ratio": 0.9,
        "dynamic_ratio": 0.1,
        "static_params": int(model_info["total_params"] * 0.9),
        "dynamic_params": int(model_info["total_params"] * 0.1)
    }
    
    split_path = MODEL_DIR / "weight_split_info.json"
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"  ✓ 权重拆分信息已保存: {split_path}")

def main():
    check_dependencies()
    model, tokenizer = download_model()
    model_info = analyze_model(model)
    save_info(model_info)
    
    print("\n" + "=" * 60)
    print("模型下载完成！")
    print("=" * 60)
    print(f"\n模型路径: {MODEL_DIR}")
    print(f"参数量: {model_info['total_params']/1e6:.1f}M")

if __name__ == "__main__":
    main()
