#!/usr/bin/env python3
"""
模块6：专项全流程训练模块
Module 6: Specialized Training Module

包含：
1. 底座预适配微调模块
2. 在线终身学习训练模块
3. 离线记忆巩固与推理优化模块
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.brain_architecture_full import (
    BrainArchitecture, BrainConfig,
    STDPSystem, HippocampusSystem,
    OptimizationMode, CyclePhase
)


# ==================== 训练数据集 ====================

class BrainTrainingDataset(Dataset):
    """训练数据集"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
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


# ==================== 子模块1：底座预适配微调 ====================

class PreAdaptationTrainer:
    """
    底座预适配微调模块
    部署前一次性执行
    """
    
    def __init__(self, brain: BrainArchitecture, config: BrainConfig):
        self.brain = brain
        self.config = config
        
        # 训练配置
        self.learning_rate = 1e-5
        self.batch_size = 8
        self.epochs = 3
        self.warmup_ratio = 0.1
        
        # 训练数据路径
        self.data_paths = {
            "alpaca": "./data/alpaca_chinese_lite.json",
            "sharegpt": "./data/sharegpt_lite.json",
            "gsm8k": "./data/gsm8k_lite.json",
            "timedial": "./data/timedial_lite.json",
            "memory": "./data/memory_recall_lite.json"
        }
    
    def load_training_data(self) -> List[Dict]:
        """加载训练数据"""
        all_data = []
        
        # 基础对话数据
        basic_data = [
            {"input": "你好", "output": "你好！我是类人脑AI助手，很高兴为您服务。"},
            {"input": "你是谁", "output": "我是基于类人脑双系统架构的AI，具有海马体记忆和STDP学习能力。"},
            {"input": "谢谢", "output": "不客气！如果还有其他问题，随时可以问我。"},
            {"input": "再见", "output": "再见！祝您生活愉快，期待下次与您交流。"},
        ]
        all_data.extend(basic_data)
        
        # 知识问答数据
        knowledge_data = [
            {"input": "什么是人工智能？", "output": "人工智能（AI）是模拟人类智能的技术，包括学习、推理、感知等能力。"},
            {"input": "什么是深度学习？", "output": "深度学习是机器学习的分支，使用多层神经网络进行特征学习和模式识别。"},
            {"input": "什么是STDP？", "output": "STDP是脉冲时序依赖可塑性，一种基于时序的突触可塑性学习规则。"},
            {"input": "什么是海马体？", "output": "海马体是大脑中负责记忆编码的关键区域，包括EC、DG、CA3、CA1等子区域。"},
            {"input": "100Hz刷新周期是什么意思？", "output": "100Hz刷新周期意味着每秒执行100次刷新，即每10毫秒完成一次推理周期。"},
        ]
        all_data.extend(knowledge_data)
        
        # 数学推理数据
        math_data = [
            {"input": "1+1等于几？", "output": "1+1=2"},
            {"input": "2+3等于几？", "output": "2+3=5"},
            {"input": "10-5等于几？", "output": "10-5=5"},
            {"input": "3乘以4等于几？", "output": "3×4=12"},
            {"input": "如果A>B，B>C，那么A和C谁大？", "output": "根据传递性，A>B且B>C，所以A>C，A比C大。"},
        ]
        all_data.extend(math_data)
        
        # 时序推理数据
        temporal_data = [
            {"input": "今天星期一，明天是星期几？", "output": "明天是星期二。"},
            {"input": "现在是上午9点，3小时后是几点？", "output": "3小时后是中午12点。"},
            {"input": "昨天是今天的前一天吗？", "output": "是的，昨天是今天的前一天。"},
        ]
        all_data.extend(temporal_data)
        
        # 记忆召回数据
        memory_data = [
            {"input": "请记住：我的名字是小明。", "output": "好的，我已经记住了您的名字是小明。"},
            {"input": "我叫什么名字？", "output": "根据之前的对话，您的名字是小明。"},
            {"input": "请记住：我喜欢蓝色。", "output": "好的，我已经记住了您喜欢蓝色。"},
            {"input": "我喜欢什么颜色？", "output": "根据之前的对话，您喜欢蓝色。"},
        ]
        all_data.extend(memory_data)
        
        return all_data
    
    def train(self) -> Dict:
        """
        执行预适配训练
        全程冻结90%静态权重，仅微调10%动态权重
        """
        print("=" * 70)
        print("底座预适配微调训练")
        print("=" * 70)
        
        # 加载数据
        print("\n[1/4] 加载训练数据...")
        training_data = self.load_training_data()
        print(f"  训练样本数: {len(training_data)}")
        
        # 创建数据集
        dataset = BrainTrainingDataset(
            training_data, 
            self.brain.tokenizer,
            max_length=512
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # 配置优化器（只优化动态权重）
        print("\n[2/4] 配置优化器...")
        dynamic_params = [
            p for n, p in self.brain.model.named_parameters() 
            if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(dynamic_params, lr=self.learning_rate)
        
        # 学习率调度
        total_steps = len(dataloader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        print(f"  可训练参数: {sum(p.numel() for p in dynamic_params):,}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  批次大小: {self.batch_size}")
        print(f"  训练轮次: {self.epochs}")
        
        # 训练循环
        print("\n[3/4] 开始训练...")
        training_log = []
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_stdp_updates = 0
            
            print(f"\nEpoch {epoch + 1}/{self.epochs}:")
            
            for batch_idx, batch in enumerate(dataloader):
                # 前向传播
                self.brain.model.train()
                outputs = self.brain.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    output_hidden_states=True
                )
                
                loss = outputs.loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # STDP更新
                if outputs.hidden_states:
                    contribution = min(1.0, torch.mean(torch.abs(outputs.hidden_states[-1])).item() / 10.0)
                    stdp_updates = self.brain.engine.stdp_system.update_weights(
                        outputs.hidden_states,
                        loss.item(),
                        contribution
                    )
                    epoch_stdp_updates += len(stdp_updates)
                
                # 梯度更新
                torch.nn.utils.clip_grad_norm_(dynamic_params, 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(dataloader)
            training_log.append({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "stdp_updates": epoch_stdp_updates
            })
            
            print(f"\n  Epoch {epoch + 1} 完成:")
            print(f"    平均损失: {avg_loss:.4f}")
            print(f"    STDP更新: {epoch_stdp_updates}")
        
        total_time = time.time() - start_time
        
        # 保存模型
        print("\n[4/4] 保存预适配权重...")
        save_path = Path("./models/pre_adapted")
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.brain.model.save_pretrained(str(save_path))
        self.brain.tokenizer.save_pretrained(str(save_path))
        
        # 保存训练日志
        log_path = save_path / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                "config": {
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs
                },
                "training_log": training_log,
                "total_time": total_time,
                "stdp_stats": self.brain.engine.stdp_system.get_statistics()
            }, f, indent=2)
        
        print(f"✓ 预适配权重已保存: {save_path}")
        
        return {
            "success": True,
            "training_log": training_log,
            "total_time": total_time,
            "save_path": str(save_path)
        }


# ==================== 子模块2：在线终身学习 ====================

class OnlineLearningSystem:
    """
    在线终身学习训练模块
    推理时实时执行
    """
    
    def __init__(self, brain: BrainArchitecture, config: BrainConfig):
        self.brain = brain
        self.config = config
        
        # 学习状态
        self.is_learning = False
        self.learning_rate = 1e-6
        
        # 统计
        self.total_updates = 0
        self.correct_predictions = 0
        self.total_predictions = 0
    
    def on_inference_complete(self, 
                               user_input: str,
                               output: str,
                               is_correct: bool = None):
        """
        推理完成后的学习回调
        基于STDP规则更新权重
        """
        if not self.is_learning:
            return
        
        # 更新统计
        self.total_predictions += 1
        if is_correct is not None:
            if is_correct:
                self.correct_predictions += 1
        
        # STDP更新已在推理周期中完成
        self.total_updates += 1
    
    def get_accuracy(self) -> float:
        """获取准确率"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "is_learning": self.is_learning,
            "total_updates": self.total_updates,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.get_accuracy()
        }


# ==================== 子模块3：离线记忆巩固 ====================

class OfflineConsolidation:
    """
    离线记忆巩固与推理优化模块
    端侧空闲时执行
    """
    
    def __init__(self, brain: BrainArchitecture, config: BrainConfig):
        self.brain = brain
        self.config = config
        
        # SWR配置
        self.swr_active = False
        self.swr_interval = 300  # 5分钟
        self.swr_duration = 60   # 60秒
        
        # 巩固统计
        self.consolidation_count = 0
        self.memories_consolidated = 0
    
    def should_consolidate(self, idle_time: float) -> bool:
        """判断是否应该执行巩固"""
        return idle_time >= self.swr_interval and not self.swr_active
    
    def start_consolidation(self):
        """启动离线巩固"""
        print("[Consolidation] 启动离线记忆巩固...")
        self.swr_active = True
        self.brain.start_swr_replay()
    
    def execute_consolidation_step(self) -> Dict:
        """执行一步巩固"""
        if not self.swr_active:
            return {"active": False}
        
        # 获取下一个记忆
        memory = self.brain.engine.hippocampus.swr_replay_step()
        
        if memory is None:
            self.swr_active = False
            self.consolidation_count += 1
            print(f"[Consolidation] 巩固完成，已巩固 {self.memories_consolidated} 条记忆")
            return {"active": False, "completed": True}
        
        # 模拟回放和权重更新
        self.memories_consolidated += 1
        
        # STDP更新（增强正确路径）
        # 这里简化实现，实际应该回放推理过程
        
        return {
            "active": True,
            "memory_id": memory.id,
            "consolidated": self.memories_consolidated
        }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "swr_active": self.swr_active,
            "consolidation_count": self.consolidation_count,
            "memories_consolidated": self.memories_consolidated
        }


# ==================== 训练管理器 ====================

class TrainingManager:
    """训练管理器 - 整合所有训练模块"""
    
    def __init__(self, brain: BrainArchitecture, config: BrainConfig):
        self.brain = brain
        self.config = config
        
        # 初始化子模块
        self.pre_adaptation = PreAdaptationTrainer(brain, config)
        self.online_learning = OnlineLearningSystem(brain, config)
        self.offline_consolidation = OfflineConsolidation(brain, config)
    
    def run_pre_adaptation(self) -> Dict:
        """运行预适配训练"""
        return self.pre_adaptation.train()
    
    def start_online_learning(self):
        """启动在线学习"""
        self.online_learning.is_learning = True
        print("[TrainingManager] 在线学习已启动")
    
    def stop_online_learning(self):
        """停止在线学习"""
        self.online_learning.is_learning = False
        print("[TrainingManager] 在线学习已停止")
    
    def check_offline_consolidation(self, idle_time: float):
        """检查并执行离线巩固"""
        if self.offline_consolidation.should_consolidate(idle_time):
            self.offline_consolidation.start_consolidation()
    
    def get_all_statistics(self) -> Dict:
        """获取所有统计信息"""
        return {
            "online_learning": self.online_learning.get_statistics(),
            "offline_consolidation": self.offline_consolidation.get_statistics()
        }


# ==================== 导出 ====================

__all__ = [
    'PreAdaptationTrainer',
    'OnlineLearningSystem',
    'OfflineConsolidation',
    'TrainingManager',
    'BrainTrainingDataset'
]
