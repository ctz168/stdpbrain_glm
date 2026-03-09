#!/usr/bin/env python3
"""下载Qwen模型"""

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = "/home/z/my-project/models/qwen3.5-0.8b"

print(f"下载模型: {MODEL_NAME}")
print(f"保存到: {MODEL_DIR}")

print("\n下载Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(MODEL_DIR)
print("✓ Tokenizer下载完成")

print("\n下载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True
)
model.save_pretrained(MODEL_DIR)
print("✓ 模型下载完成")

print("\n验证...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
print("✓ 验证通过")

total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数: {total_params/1e6:.1f}M")
