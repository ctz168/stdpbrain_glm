#!/usr/bin/env python3
"""
快速推理脚本 - 用于API调用
"""

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 获取参数
prompt = sys.argv[1] if len(sys.argv) > 1 else "你好"
model_dir = os.environ.get('MODEL_DIR', '/home/z/my-project/models/qwen3.5-0.8b')

try:
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # 推理
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
    print(response)
    
except Exception as e:
    print(f"错误: {e}")
    sys.exit(1)
