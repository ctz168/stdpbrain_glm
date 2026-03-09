#!/usr/bin/env python3
"""
下载Qwen3.5-0.8B-Base模型
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"
MODEL_DIR = "/home/z/my-project/models/qwen3.5-0.8b"

print("=" * 60)
print("下载Qwen3.5-0.8B-Base模型")
print("=" * 60)
print(f"模型: {MODEL_NAME}")
print(f"保存到: {MODEL_DIR}")
print()

# 清理旧模型
if os.path.exists(MODEL_DIR):
    import shutil
    print("清理旧模型...")
    shutil.rmtree(MODEL_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n[1/3] 下载Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(MODEL_DIR)
print("✓ Tokenizer下载完成")

print("\n[2/3] 下载模型权重...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True
)
model.save_pretrained(MODEL_DIR)
print("✓ 模型下载完成")

print("\n[3/3] 验证模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
print("✓ 验证通过")

total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数: {total_params/1e6:.1f}M")
print(f"隐藏层大小: {model.config.hidden_size}")
print(f"层数: {model.config.num_hidden_layers}")
print(f"注意力头数: {model.config.num_attention_heads}")

# 保存模型信息
import json
info = {
    "model_name": MODEL_NAME,
    "total_params": total_params,
    "hidden_size": model.config.hidden_size,
    "num_layers": model.config.num_hidden_layers,
    "num_attention_heads": model.config.num_attention_heads,
    "vocab_size": model.config.vocab_size
}
with open(f"{MODEL_DIR}/model_info.json", "w") as f:
    json.dump(info, f, indent=2)

print("\n✓ 模型下载完成!")
