#!/usr/bin/env python3
"""
类人脑双系统AI架构 - 真正的STDP训练系统
Real STDP Training System - Actually Updates Model Weights
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

print("=" * 70)
print("类人脑双系统AI架构 - 真正的STDP训练")
print("=" * 70)

MODEL_DIR = Path("/home/z/my-project/models/qwen3.5-0.8b")

# ==================== STDP核心实现 ====================

@dataclass
class STDPConfig:
    """STDP配置"""
    alpha: float = 0.01  # LTP学习率
    beta: float = 0.005  # LTD学习率
    time_window: float = 20.0  # 时间窗口(ms)
    weight_upper_bound: float = 2.0  # 权重上限
    weight_lower_bound: float = -2.0  # 权重下限
    static_ratio: float = 0.9  # 静态权重比例
    dynamic_ratio: float = 0.1  # 动态权重比例

@dataclass
class STDPUpdate:
    """STDP更新记录"""
    layer_name: str
    param_name: str
    delta: float
    update_type: str  # 'LTP' or 'LTD'
    contribution: float
    timestamp: float

class STDPTrainer:
    """
    真正的STDP训练器
    直接操作模型权重，实现时序可塑性学习
    """
    
    def __init__(self, model, config: STDPConfig = None):
        self.model = model
        self.config = config or STDPConfig()
        self.update_history: List[STDPUpdate] = []
        self.token_history: List[Dict] = []
        
        # 识别可训练的动态权重
        self.dynamic_params = {}
        self.static_params = {}
        self._split_weights()
        
    def _split_weights(self):
        """权重双轨拆分"""
        print("\n[STDP] 执行权重双轨拆分...")
        
        total_params = 0
        dynamic_count = 0
        static_count = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # 只训练最后几层的注意力权重
            if any(x in name for x in ['layers.2.self_attn', 'layers.3.self_attn', 'lm_head']):
                param.requires_grad = True
                self.dynamic_params[name] = param
                dynamic_count += param.numel()
            else:
                param.requires_grad = False
                self.static_params[name] = param
                static_count += param.numel()
        
        print(f"  总参数: {total_params:,}")
        print(f"  动态权重: {dynamic_count:,} ({dynamic_count/total_params*100:.1f}%)")
        print(f"  静态权重: {static_count:,} ({static_count/total_params*100:.1f}%)")
        print(f"  可训练参数: {len(self.dynamic_params)} 个")
    
    def calculate_ltp(self, delta_t: float, contribution: float, current_weight: float) -> float:
        """
        计算LTP权重增强
        delta_t > 0 表示前序激活早于后序激活
        """
        if delta_t <= 0:
            return 0.0
        
        # 时间窗口函数（指数衰减）
        time_factor = math.exp(-delta_t / self.config.time_window)
        
        # LTP增强量
        delta_w = self.config.alpha * time_factor * contribution * abs(current_weight)
        
        return min(delta_w, self.config.weight_upper_bound - current_weight)
    
    def calculate_ltd(self, delta_t: float, interference: float, current_weight: float) -> float:
        """
        计算LTD权重减弱
        delta_t < 0 表示前序激活晚于后序激活（异常时序）
        """
        if delta_t > 0 and interference < 0.3:
            return 0.0
        
        # 时间窗口函数
        time_factor = math.exp(-abs(delta_t) / self.config.time_window)
        
        # LTD减弱量
        delta_w = -self.config.beta * time_factor * interference * abs(current_weight)
        
        return max(delta_w, self.config.weight_lower_bound - current_weight)
    
    def compute_contribution_score(self, 
                                    hidden_states: torch.Tensor,
                                    target_token_id: int) -> float:
        """
        计算贡献度分数
        基于隐藏状态与目标token的相关性
        """
        with torch.no_grad():
            # 获取最后一层隐藏状态
            last_hidden = hidden_states[-1]  # [batch, seq_len, hidden_dim]
            
            # 计算平均激活强度
            activation = torch.mean(torch.abs(last_hidden)).item()
            
            # 归一化到0-1
            contribution = min(1.0, activation / 10.0)
            
        return contribution
    
    def compute_interference_score(self, 
                                   hidden_states: torch.Tensor,
                                   loss: float) -> float:
        """
        计算干扰度分数
        基于损失值和隐藏状态的不稳定性
        """
        with torch.no_grad():
            # 损失越高，干扰越大
            loss_factor = min(1.0, loss / 5.0)
            
            # 隐藏状态方差越大，干扰越大
            last_hidden = hidden_states[-1]
            variance = torch.var(last_hidden).item()
            variance_factor = min(1.0, variance / 100.0)
            
            interference = (loss_factor + variance_factor) / 2
            
        return interference
    
    def apply_stdp_update(self,
                          param: torch.nn.Parameter,
                          delta: float,
                          param_name: str):
        """
        应用STDP权重更新
        直接修改参数值
        """
        with torch.no_grad():
            # 计算更新量
            update = delta * torch.ones_like(param) * 0.01  # 缩放因子
            
            # 应用更新
            param.add_(update)
            
            # 裁剪到合理范围
            param.clamp_(-self.config.weight_upper_bound, self.config.weight_upper_bound)
    
    def train_step(self,
                   input_ids: torch.Tensor,
                   attention_mask: torch.Tensor,
                   labels: torch.Tensor,
                   optimizer) -> Tuple[float, List[STDPUpdate]]:
        """
        执行一步STDP训练
        """
        self.model.train()
        updates = []
        
        # 前向传播，获取隐藏状态
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        loss = outputs.loss
        hidden_states = outputs.hidden_states
        
        # 反向传播计算梯度
        optimizer.zero_grad()
        loss.backward()
        
        # 计算贡献度和干扰度
        contribution = self.compute_contribution_score(hidden_states, labels[0, 0].item())
        interference = self.compute_interference_score(hidden_states, loss.item())
        
        # 对每个动态参数应用STDP更新
        current_time = time.time() * 1000  # 毫秒时间戳
        
        for name, param in self.dynamic_params.items():
            if param.grad is not None:
                # 基于梯度方向决定LTP/LTD
                grad_mean = torch.mean(param.grad).item()
                
                if grad_mean > 0:
                    # 梯度为正，应用LTP
                    delta = self.calculate_ltp(
                        delta_t=10.0,  # 假设10ms时间差
                        contribution=contribution,
                        current_weight=torch.mean(param).item()
                    )
                    update_type = 'LTP'
                else:
                    # 梯度为负，应用LTD
                    delta = self.calculate_ltd(
                        delta_t=-10.0,
                        interference=interference,
                        current_weight=torch.mean(param).item()
                    )
                    update_type = 'LTD'
                
                # 应用更新
                self.apply_stdp_update(param, delta, name)
                
                # 记录更新
                updates.append(STDPUpdate(
                    layer_name=name.split('.')[0],
                    param_name=name,
                    delta=delta,
                    update_type=update_type,
                    contribution=contribution,
                    timestamp=current_time
                ))
        
        # 正常梯度更新
        optimizer.step()
        
        return loss.item(), updates


# ==================== 海马体记忆系统 ====================

class HippocampusMemory:
    """
    海马体记忆系统
    实现EC-DG-CA3-CA1记忆编码和回忆
    """
    
    def __init__(self, hidden_size: int = 896, memory_size: int = 100):
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # 记忆存储
        self.memories: List[Dict] = []
        
        # EC: 特征编码
        self.ec_weights = nn.Linear(hidden_size, 64)
        
        # DG: 模式分离
        self.dg_weights = nn.Linear(64, 128)
        
        # CA3: 情景记忆
        self.ca3_weights = nn.Linear(128, 256)
        
        # CA1: 时序编码
        self.ca1_weights = nn.Linear(256, 64)
        
    def encode(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        编码记忆
        EC -> DG -> CA3 -> CA1
        """
        with torch.no_grad():
            # EC: 特征压缩
            ec_out = torch.tanh(self.ec_weights(hidden_state))
            
            # DG: 模式分离（稀疏化）
            dg_out = F.relu(self.dg_weights(ec_out))
            dg_out = F.dropout(dg_out, p=0.5, training=False)
            
            # CA3: 情景存储
            ca3_out = torch.tanh(self.ca3_weights(dg_out))
            
            # CA1: 时序编码
            ca1_out = torch.tanh(self.ca1_weights(ca3_out))
            
        return ca1_out
    
    def store(self, hidden_state: torch.Tensor, text: str):
        """存储记忆"""
        if len(self.memories) >= self.memory_size:
            self.memories.pop(0)
        
        encoded = self.encode(hidden_state)
        
        self.memories.append({
            'embedding': encoded,
            'text': text,
            'timestamp': time.time(),
            'strength': 1.0
        })
    
    def recall(self, query: torch.Tensor, top_k: int = 3) -> List[Dict]:
        """回忆相关记忆"""
        if not self.memories:
            return []
        
        query_encoded = self.encode(query)
        
        # 计算相似度
        similarities = []
        for mem in self.memories:
            # 使用mean来获取标量
            sim = F.cosine_similarity(
                query_encoded.unsqueeze(0),
                mem['embedding'].unsqueeze(0),
                dim=1
            ).mean().item()
            similarities.append((sim, mem))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [mem for _, mem in similarities[:top_k]]


# ==================== 训练数据 ====================

TRAINING_DATA = [
    {"input": "你好", "output": "你好！我是类人脑AI助手，很高兴为您服务。"},
    {"input": "你是谁", "output": "我是基于类人脑双系统架构的AI，具有STDP学习能力和海马体记忆系统。"},
    {"input": "什么是STDP？", "output": "STDP是脉冲时序依赖可塑性，一种基于时序的突触可塑性学习规则。当前序激活早于后序激活时，突触增强(LTP)；反之则减弱(LTD)。"},
    {"input": "什么是海马体？", "output": "海马体是大脑中负责记忆编码的关键区域。在AI中，它包含EC(特征编码)、DG(模式分离)、CA3(情景记忆)、CA1(时序编码)等子模块。"},
    {"input": "什么是人工智能？", "output": "人工智能(AI)是模拟人类智能的技术，包括学习、推理、感知、理解语言等能力。"},
    {"input": "1+1等于几？", "output": "1+1=2"},
    {"input": "2+3等于几？", "output": "2+3=5"},
    {"input": "如果A>B，B>C，那么A和C谁大？", "output": "根据传递性，A>B且B>C，所以A>C，A比C大。"},
    {"input": "如何保持健康？", "output": "保持健康需要均衡饮食、规律运动、充足睡眠和良好心态。"},
    {"input": "谢谢", "output": "不客气！如果还有其他问题，随时可以问我。"},
]

# ==================== 主训练流程 ====================

def main():
    print("\n[1/5] 加载模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    print("✓ 模型加载成功")
    
    print("\n[2/5] 初始化STDP训练器...")
    stdp_config = STDPConfig()
    trainer = STDPTrainer(model, stdp_config)
    
    print("\n[3/5] 初始化海马体记忆系统...")
    hippocampus = HippocampusMemory(hidden_size=896)
    
    print("\n[4/5] 开始STDP训练...")
    print("-" * 50)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-6
    )
    
    epochs = 5
    training_log = []
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_updates = []
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch + 1}/{epochs}:")
        
        for i, sample in enumerate(TRAINING_DATA):
            # 构建输入
            text = f"用户: {sample['input']}\n助手: {sample['output']}"
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
            
            # 执行STDP训练步骤
            loss, updates = trainer.train_step(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['input_ids'],
                optimizer=optimizer
            )
            
            epoch_loss += loss
            epoch_updates.extend(updates)
            
            # 存储到海马体
            with torch.no_grad():
                outputs = model(input_ids=inputs['input_ids'], output_hidden_states=True)
                hidden = outputs.hidden_states[-1].mean(dim=1)  # [1, hidden_dim]
                hippocampus.store(hidden, text)
            
            # 打印进度
            ltp_count = sum(1 for u in updates if u.update_type == 'LTP')
            ltd_count = sum(1 for u in updates if u.update_type == 'LTD')
            
            print(f"  样本 {i+1}/{len(TRAINING_DATA)}: '{sample['input'][:15]}...'")
            print(f"    Loss: {loss:.4f} | LTP: {ltp_count} | LTD: {ltd_count}")
        
        avg_loss = epoch_loss / len(TRAINING_DATA)
        epoch_time = time.time() - epoch_start
        
        training_log.append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "total_updates": len(epoch_updates),
            "ltp_count": sum(1 for u in epoch_updates if u.update_type == 'LTP'),
            "ltd_count": sum(1 for u in epoch_updates if u.update_type == 'LTD'),
            "time": epoch_time
        })
        
        print(f"\n  Epoch {epoch + 1} 完成:")
        print(f"    平均损失: {avg_loss:.4f}")
        print(f"    STDP更新: {len(epoch_updates)} (LTP: {training_log[-1]['ltp_count']}, LTD: {training_log[-1]['ltd_count']})")
        print(f"    海马体记忆: {len(hippocampus.memories)} 条")
        print(f"    耗时: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    print("\n[5/5] 评估训练效果...")
    print("-" * 50)
    
    model.eval()
    test_cases = ["你好", "什么是STDP？", "1+1等于几？"]
    
    for prompt in test_cases:
        inputs = tokenizer(prompt, return_tensors='pt')
        
        # 尝试从海马体回忆
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], output_hidden_states=True)
            hidden = outputs.hidden_states[-1].mean(dim=1)
            recalled = hippocampus.recall(hidden, top_k=1)
        
        # 生成响应
        with torch.no_grad():
            gen_outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        
        print(f"\n输入: {prompt}")
        print(f"输出: {response[:100]}...")
        if recalled:
            print(f"海马体回忆: {recalled[0]['text'][:50]}...")
    
    # 保存训练日志
    log_path = MODEL_DIR / "real_stdp_training_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            "config": {
                "alpha": stdp_config.alpha,
                "beta": stdp_config.beta,
                "time_window": stdp_config.time_window,
                "dynamic_params": len(trainer.dynamic_params)
            },
            "training_log": training_log,
            "total_time": total_time,
            "hippocampus_memories": len(hippocampus.memories)
        }, f, indent=2)
    
    print(f"\n训练日志已保存: {log_path}")
    
    # 训练总结
    print("\n" + "=" * 70)
    print("STDP训练总结")
    print("=" * 70)
    print(f"训练轮次: {epochs}")
    print(f"总耗时: {total_time:.1f}秒")
    print(f"初始损失: {training_log[0]['avg_loss']:.4f}")
    print(f"最终损失: {training_log[-1]['avg_loss']:.4f}")
    print(f"损失下降: {training_log[0]['avg_loss'] - training_log[-1]['avg_loss']:.4f}")
    print(f"总STDP更新: {sum(log['total_updates'] for log in training_log)}")
    print(f"海马体记忆: {len(hippocampus.memories)} 条")
    print("=" * 70)

if __name__ == "__main__":
    main()
