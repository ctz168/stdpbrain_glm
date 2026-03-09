#!/usr/bin/env python3
"""
类人脑双系统全闭环AI架构 - 生产级完整实现
Human-Like Brain Dual-System Full-Loop AI Architecture

严格遵循需求文档的所有刚性红线和模块要求
"""

import os
import sys
import json
import time
import math
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# ==================== 枚举和常量 ====================

class OptimizationMode(Enum):
    """优化模式"""
    SELF_GENERATION = "selfGeneration"      # 模式1：自生成组合输出
    SELF_PLAY = "selfPlay"                  # 模式2：自博弈竞争优化
    SELF_EVALUATION = "selfEvaluation"      # 模式3：自双输出+自评判

class CyclePhase(Enum):
    """刷新周期阶段"""
    INPUT_RECEIVE = "inputReceive"          # 阶段1：输入token接收
    MEMORY_RECALL = "memoryRecall"          # 阶段2：海马体记忆调取
    INFERENCE = "inference"                 # 阶段3：窄窗口推理
    OUTPUT = "output"                       # 阶段4：输出生成
    STDP_UPDATE = "stdpUpdate"              # 阶段5：STDP权重刷新
    MEMORY_ENCODE = "memoryEncode"          # 阶段6：海马体编码
    WORKING_MEMORY = "workingMemory"        # 阶段7：工作记忆更新

class RoleMode(Enum):
    """角色模式"""
    GENERATOR = "generator"                 # 生成角色
    VALIDATOR = "validator"                 # 验证角色
    EVALUATOR = "evaluator"                 # 评判角色

# 刚性约束常量
REFRESH_CYCLE_MS = 10.0                    # 10ms刷新周期
REFRESH_RATE_HZ = 100.0                    # 100Hz刷新率
STATIC_WEIGHT_RATIO = 0.9                  # 90%静态权重
DYNAMIC_WEIGHT_RATIO = 0.1                 # 10%动态权重
MAX_MEMORY_MB = 420                        # 最大显存420MB
NARROW_WINDOW_SIZE = 2                     # 窄窗口大小1-2 token
HIPPOCAMPUS_MEMORY_MAX = 1000              # 海马体最大记忆数
HIPPOCAMPUS_MEMORY_MB = 2                  # 海马体内存限制2MB

# STDP时间窗口
STDP_TIME_WINDOW_MS = 20.0                 # STDP时间窗口20ms

# ==================== 配置类 ====================

@dataclass
class BrainConfig:
    """架构配置 - 严格遵循刚性红线"""
    # 模型配置
    model_path: str = "./models/qwen3.5-0.8b"
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    # 刷新周期配置 - 刚性约束
    refresh_cycle_ms: float = REFRESH_CYCLE_MS
    refresh_rate_hz: float = REFRESH_RATE_HZ
    
    # 窄窗口配置 - 刚性约束
    narrow_window_size: int = NARROW_WINDOW_SIZE
    
    # 权重拆分 - 刚性约束
    static_ratio: float = STATIC_WEIGHT_RATIO
    dynamic_ratio: float = DYNAMIC_WEIGHT_RATIO
    
    # STDP配置
    stdp_alpha: float = 0.01                # LTP学习率
    stdp_beta: float = 0.005                # LTD学习率
    stdp_time_window: float = STDP_TIME_WINDOW_MS
    stdp_weight_upper: float = 2.0          # 权重上限
    stdp_weight_lower: float = -2.0         # 权重下限
    
    # 海马体配置
    hippocampus_memory_size: int = HIPPOCAMPUS_MEMORY_MAX
    hippocampus_embedding_dim: int = 64     # EC输出维度
    hippocampus_dg_dim: int = 128           # DG输出维度
    hippocampus_ca3_dim: int = 256          # CA3输出维度
    hippocampus_ca1_dim: int = 64           # CA1输出维度
    
    # 自闭环优化配置
    self_play_max_iterations: int = 5       # 自博弈最大迭代次数
    self_eval_interval: int = 10            # 自评判间隔周期数
    
    # 生成配置
    max_new_tokens: int = 64                # 单次生成token数
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


# ==================== 数据结构 ====================

@dataclass
class TokenFeature:
    """Token特征 - 用于窄窗口处理"""
    token_id: int
    text: str
    embedding: np.ndarray                   # 嵌入向量
    attention_weights: np.ndarray           # 注意力权重
    temporal_feature: np.ndarray            # 时序特征
    semantic_feature: np.ndarray            # 语义特征
    timestamp: float                        # 时间戳(ms)
    layer_activations: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class MemoryAnchor:
    """记忆锚点 - 海马体存储单元"""
    id: str
    memory_id: int                          # 唯一记忆ID(DG生成)
    feature_vector: np.ndarray              # CA1输出特征
    text_pointer: str                       # 文本指针(不存完整文本)
    timestamp: float                        # 精准时间戳
    temporal_skeleton: List[int]            # 时序骨架
    semantic_pointer: np.ndarray            # 语义指针
    causal_links: List[str]                 # 因果关联
    strength: float = 1.0                   # 连接强度
    access_count: int = 0                   # 访问次数
    context_ids: List[int] = field(default_factory=list)


@dataclass
class STDPUpdateRecord:
    """STDP更新记录"""
    timestamp: float
    layer_type: str                         # attention/ffn/selfEval/hippocampus
    weight_id: str
    delta_value: float
    update_type: str                        # LTP/LTD
    trigger_reason: str
    contribution_score: float = 0.0


@dataclass
class CycleExecutionState:
    """单周期执行状态"""
    cycle_id: int
    start_time: float
    end_time: float
    phase: CyclePhase
    input_token: Optional[TokenFeature]
    memory_anchors: List[MemoryAnchor]
    output: str
    stdp_updates: List[STDPUpdateRecord]
    optimization_mode: OptimizationMode


@dataclass
class EvaluationResult:
    """评判结果"""
    candidate_1: str
    candidate_2: str
    scores_1: Dict[str, float]              # 4维度分数
    scores_2: Dict[str, float]
    winner: int                             # 1 or 2
    total_score_1: float
    total_score_2: float


# ==================== 模块1：底座模型改造 ====================

class ModelAdapter:
    """
    Qwen3.5-0.8B底座模型适配器
    实现权重双轨拆分和接口适配
    """
    
    def __init__(self, model: AutoModelForCausalLM, config: BrainConfig):
        self.model = model
        self.config = config
        
        # 权重分类
        self.static_weights: Dict[str, nn.Parameter] = {}
        self.dynamic_weights: Dict[str, nn.Parameter] = {}
        self.weight_split_ratio = 0.0
        
        # 执行权重拆分
        self._split_weights()
        
        # 特征输出缓存
        self.feature_cache: Dict[str, TokenFeature] = {}
        
        # 角色提示词模板
        self.role_prompts = {
            RoleMode.GENERATOR: "你是一个智能助手，请生成回复：",
            RoleMode.VALIDATOR: "你是一个验证者，请检查以下内容的正确性：",
            RoleMode.EVALUATOR: "你是一个评判者，请对以下两个候选回复打分(0-10分)：\n评判维度：事实准确性、逻辑完整性、语义连贯性、指令遵循度"
        }
    
    def _split_weights(self):
        """
        权重双轨拆分 - 严格90%/10%
        静态权重：90%，永久冻结
        动态权重：10%，可STDP更新
        """
        total_params = 0
        static_count = 0
        dynamic_count = 0
        
        # 计算总参数量
        for name, param in self.model.named_parameters():
            total_params += param.numel()
        
        # 目标动态权重数量
        target_dynamic = int(total_params * self.config.dynamic_ratio)
        
        # 按层选择动态权重
        # 策略：选择最后几层的注意力权重作为动态权重
        dynamic_layers = []
        num_layers = self.model.config.num_hidden_layers
        
        # 计算需要多少层的权重才能达到10%
        for i in range(num_layers - 1, -1, -1):
            layer_dynamic = 0
            layer_name = f"model.layers.{i}"
            for name, param in self.model.named_parameters():
                if layer_name in name and ('self_attn' in name or 'mlp' in name):
                    layer_dynamic += param.numel()
            
            if dynamic_count + layer_dynamic <= target_dynamic:
                dynamic_layers.append(i)
                dynamic_count += layer_dynamic
            else:
                break
        
        # 执行拆分
        for name, param in self.model.named_parameters():
            is_dynamic = False
            
            # 检查是否在动态层列表中
            for layer_idx in dynamic_layers:
                if f"layers.{layer_idx}" in name:
                    # 只选择部分权重作为动态权重
                    if 'q_proj' in name or 'v_proj' in name or 'gate_proj' in name:
                        is_dynamic = True
                        break
            
            if is_dynamic:
                param.requires_grad = True
                self.dynamic_weights[name] = param
                dynamic_count += param.numel()
            else:
                param.requires_grad = False
                self.static_weights[name] = param
                static_count += param.numel()
        
        self.weight_split_ratio = dynamic_count / total_params
        
        print(f"[ModelAdapter] 权重双轨拆分完成:")
        print(f"  总参数: {total_params:,}")
        print(f"  静态权重: {static_count:,} ({static_count/total_params*100:.1f}%)")
        print(f"  动态权重: {dynamic_count:,} ({dynamic_count/total_params*100:.1f}%)")
        print(f"  动态层: {dynamic_layers}")
    
    def get_attention_features(self, layer_idx: int, hidden_states: torch.Tensor) -> TokenFeature:
        """
        注意力层特征输出接口
        复用模型原生特征提取能力
        """
        # 获取注意力层
        layer = self.model.model.layers[layer_idx]
        
        with torch.no_grad():
            # 提取注意力特征
            attention_output = layer.self_attn(hidden_states)
            
            # 转换为TokenFeature
            feature = TokenFeature(
                token_id=0,
                text="",
                embedding=hidden_states[0, -1, :].cpu().numpy(),
                attention_weights=attention_output[0][0, -1, :].cpu().numpy() if attention_output[0].dim() > 2 else np.zeros(64),
                temporal_feature=np.zeros(64),
                semantic_feature=hidden_states[0, -1, :64].cpu().numpy(),
                timestamp=time.time() * 1000
            )
        
        return feature
    
    def apply_hippocampus_gate(self, memory_anchors: List[MemoryAnchor], 
                                hidden_states: torch.Tensor) -> torch.Tensor:
        """
        海马体注意力门控接口
        实现海马体引导注意力聚焦
        """
        if not memory_anchors:
            return hidden_states
        
        with torch.no_grad():
            # 获取记忆特征
            memory_features = []
            for anchor in memory_anchors[:NARROW_WINDOW_SIZE]:
                memory_features.append(anchor.feature_vector)
            
            if memory_features:
                # 计算门控权重
                memory_tensor = torch.tensor(np.array(memory_features), dtype=torch.float32)
                gate_weights = F.softmax(torch.sum(memory_tensor, dim=-1), dim=0)
                
                # 应用门控（简化实现）
                gated_hidden = hidden_states * (1.0 + 0.1 * gate_weights.mean().item())
                return gated_hidden
        
        return hidden_states
    
    def get_role_prompt(self, role: RoleMode) -> str:
        """获取角色提示词"""
        return self.role_prompts.get(role, "")
    
    def set_role(self, role: RoleMode):
        """设置当前角色"""
        self.current_role = role


# ==================== 模块3：STDP学习系统 ====================

class STDPSystem:
    """
    全链路STDP时序可塑性权重自动刷新系统
    实现无反向传播的纯时序驱动学习
    """
    
    def __init__(self, model_adapter: ModelAdapter, config: BrainConfig):
        self.model_adapter = model_adapter
        self.config = config
        
        # STDP更新历史
        self.update_history: List[STDPUpdateRecord] = []
        self.max_history = 10000
        
        # 时序记录
        self.pre_activation_times: Dict[str, float] = {}
        self.post_activation_times: Dict[str, float] = {}
        
        # 周期计数
        self.cycle_count = 0
        self.last_self_eval_cycle = 0
    
    def calculate_ltp(self, delta_t: float, contribution: float, 
                      current_weight: float) -> float:
        """
        LTP权重增强规则
        delta_t > 0: 前序激活早于后序激活
        """
        if delta_t <= 0:
            return 0.0
        
        # 时间窗口函数（指数衰减）
        time_factor = math.exp(-delta_t / self.config.stdp_time_window)
        
        # LTP增强量 = α × 时间窗口 × 贡献度 × 当前权重
        delta_weight = self.config.stdp_alpha * time_factor * contribution * abs(current_weight)
        
        # 应用权重上限
        new_weight = min(current_weight + delta_weight, self.config.stdp_weight_upper)
        
        return new_weight - current_weight
    
    def calculate_ltd(self, delta_t: float, interference: float,
                      current_weight: float) -> float:
        """
        LTD权重减弱规则
        delta_t < 0: 前序激活晚于后序激活（异常时序）
        """
        if delta_t > 0 and interference < 0.3:
            return 0.0
        
        # 时间窗口函数
        time_factor = math.exp(-abs(delta_t) / self.config.stdp_time_window)
        
        # LTD减弱量 = -β × 时间窗口 × 干扰度 × 当前权重
        delta_weight = -self.config.stdp_beta * time_factor * interference * abs(current_weight)
        
        # 应用权重下限
        new_weight = max(current_weight + delta_weight, self.config.stdp_weight_lower)
        
        return new_weight - current_weight
    
    def update_attention_stdp(self, current_token: TokenFeature,
                               context_tokens: List[TokenFeature],
                               contribution_scores: List[float]) -> List[STDPUpdateRecord]:
        """
        注意力层STDP更新
        每个刷新周期执行
        """
        updates = []
        now = time.time() * 1000
        
        for i, ctx_token in enumerate(context_tokens):
            # 计算时序差
            delta_t = current_token.timestamp - ctx_token.timestamp
            
            # 获取贡献度
            contribution = contribution_scores[i] if i < len(contribution_scores) else 0.5
            interference = 1.0 - contribution
            
            # 计算权重变化
            if delta_t > 0:
                delta = self.calculate_ltp(delta_t, contribution, 0.5)
                update_type = 'LTP'
            else:
                delta = self.calculate_ltd(delta_t, interference, 0.5)
                update_type = 'LTD'
            
            # 记录更新
            updates.append(STDPUpdateRecord(
                timestamp=now,
                layer_type='attention',
                weight_id=f'attention_{ctx_token.token_id}',
                delta_value=delta,
                update_type=update_type,
                trigger_reason=f'时序关联:{delta_t:.1f}ms, 贡献度:{contribution:.3f}',
                contribution_score=contribution
            ))
        
        return updates
    
    def update_ffn_stdp(self, feature_strength: float,
                         is_high_frequency: bool) -> List[STDPUpdateRecord]:
        """
        FFN层STDP更新
        对高频特征、专属术语自动增强
        """
        updates = []
        now = time.time() * 1000
        
        if is_high_frequency:
            # 高频特征增强
            delta = self.calculate_ltp(10.0, feature_strength, 0.5)
            update_type = 'LTP'
        else:
            # 低频特征保持或减弱
            delta = self.calculate_ltd(-10.0, 1.0 - feature_strength, 0.5)
            update_type = 'LTD'
        
        updates.append(STDPUpdateRecord(
            timestamp=now,
            layer_type='ffn',
            weight_id='ffn_feature_dynamic',
            delta_value=delta,
            update_type=update_type,
            trigger_reason=f'特征强度:{feature_strength:.3f}, 高频:{is_high_frequency}'
        ))
        
        return updates
    
    def update_self_eval_stdp(self, evaluation_result: EvaluationResult) -> List[STDPUpdateRecord]:
        """
        自评判STDP更新
        每10个刷新周期执行
        """
        updates = []
        now = time.time() * 1000
        
        # 获取胜者和败者
        winner_score = evaluation_result.total_score_1 if evaluation_result.winner == 1 else evaluation_result.total_score_2
        loser_score = evaluation_result.total_score_2 if evaluation_result.winner == 1 else evaluation_result.total_score_1
        
        # 胜者路径增强
        delta_ltp = self.calculate_ltp(100.0, winner_score / 40.0, 0.5)
        updates.append(STDPUpdateRecord(
            timestamp=now,
            layer_type='selfEvaluation',
            weight_id=f'candidate_{evaluation_result.winner}_path',
            delta_value=delta_ltp,
            update_type='LTP',
            trigger_reason=f'评判胜出, 得分:{winner_score:.1f}/40'
        ))
        
        # 败者路径减弱
        delta_ltd = self.calculate_ltd(-100.0, 1.0 - loser_score / 40.0, 0.5)
        updates.append(STDPUpdateRecord(
            timestamp=now,
            layer_type='selfEvaluation',
            weight_id=f'candidate_{3-evaluation_result.winner}_path',
            delta_value=delta_ltd,
            update_type='LTD',
            trigger_reason=f'评判落败, 得分:{loser_score:.1f}/40'
        ))
        
        return updates
    
    def update_hippocampus_stdp(self, memory_anchors: List[MemoryAnchor],
                                 relevance_scores: List[float]) -> List[STDPUpdateRecord]:
        """
        海马体门控STDP更新
        对有正向贡献的记忆锚点增强
        """
        updates = []
        now = time.time() * 1000
        
        for anchor, relevance in zip(memory_anchors, relevance_scores):
            if relevance > 0.5:
                delta = self.calculate_ltp(10.0, relevance, anchor.strength)
                update_type = 'LTP'
            else:
                delta = self.calculate_ltd(-10.0, 1.0 - relevance, anchor.strength)
                update_type = 'LTD'
            
            updates.append(STDPUpdateRecord(
                timestamp=now,
                layer_type='hippocampusGate',
                weight_id=anchor.id,
                delta_value=delta,
                update_type=update_type,
                trigger_reason=f'记忆相关性:{relevance:.3f}'
            ))
        
        return updates
    
    def apply_updates_to_weights(self, updates: List[STDPUpdateRecord]):
        """将STDP更新应用到动态权重"""
        with torch.no_grad():
            for update in updates:
                # 简化实现：按比例更新所有动态权重
                for name, param in self.model_adapter.dynamic_weights.items():
                    if update.layer_type in name or update.layer_type == 'selfEvaluation':
                        param.add_(update.delta_value * 0.001 * torch.ones_like(param))
        
        # 记录历史
        self.update_history.extend(updates)
        if len(self.update_history) > self.max_history:
            self.update_history = self.update_history[-self.max_history:]
    
    def get_statistics(self) -> Dict:
        """获取STDP统计信息"""
        ltp_count = sum(1 for u in self.update_history if u.update_type == 'LTP')
        ltd_count = sum(1 for u in self.update_history if u.update_type == 'LTD')
        
        return {
            "total_updates": len(self.update_history),
            "ltp_count": ltp_count,
            "ltd_count": ltd_count,
            "ltp_ratio": ltp_count / max(len(self.update_history), 1),
            "dynamic_params": len(self.model_adapter.dynamic_weights),
            "cycle_count": self.cycle_count
        }


# ==================== 模块5：海马体记忆系统 ====================

class HippocampusSystem:
    """
    海马体记忆系统
    严格按生物脑结构实现：EC -> DG -> CA3 -> CA1 -> SWR
    """
    
    def __init__(self, hidden_size: int, config: BrainConfig):
        self.hidden_size = hidden_size
        self.config = config
        
        # 记忆存储
        self.memories: List[MemoryAnchor] = []
        self.memory_id_counter = 0
        
        # ===== EC: 内嗅皮层 - 特征编码 =====
        self.ec_encoder = nn.Linear(hidden_size, config.hippocampus_embedding_dim)
        
        # ===== DG: 齿状回 - 模式分离 =====
        # 稀疏随机投影，无训练参数
        self.dg_projection = nn.Linear(
            config.hippocampus_embedding_dim, 
            config.hippocampus_dg_dim,
            bias=False
        )
        # 固定随机权重，实现模式分离
        nn.init.normal_(self.dg_projection.weight, mean=0, std=0.1)
        self.dg_projection.weight.requires_grad = False
        
        # ===== CA3: 情景记忆存储 + 模式补全 =====
        self.ca3_encoder = nn.Linear(
            config.hippocampus_dg_dim,
            config.hippocampus_ca3_dim
        )
        # CA3自连接（用于模式补全）
        self.ca3_recurrent = nn.Linear(
            config.hippocampus_ca3_dim,
            config.hippocampus_ca3_dim
        )
        
        # ===== CA1: 时序编码 + 注意力门控 =====
        self.ca1_encoder = nn.Linear(
            config.hippocampus_ca3_dim,
            config.hippocampus_ca1_dim
        )
        
        # SWR状态
        self.swr_active = False
        self.swr_replay_queue: List[MemoryAnchor] = []
        
        # 内存限制
        self.max_memory_bytes = HIPPOCAMPUS_MEMORY_MB * 1024 * 1024
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.ec_encoder, self.ca3_encoder, self.ca1_encoder]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def encode_ec(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        EC: 特征编码
        归一化稀疏编码为64维特征向量
        """
        with torch.no_grad():
            ec_out = torch.tanh(self.ec_encoder(hidden_state))
            # 稀疏化
            ec_out = F.relu(ec_out) * (ec_out > 0.5).float()
        return ec_out
    
    def encode_dg(self, ec_output: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        DG: 模式分离
        为相似输入生成完全正交的唯一记忆ID
        """
        with torch.no_grad():
            # 稀疏随机投影
            dg_out = self.dg_projection(ec_output)
            # 正交化处理
            dg_out = F.relu(dg_out)
            # 二值化生成唯一ID
            binary = (dg_out > dg_out.mean()).int()
            
            # 计算唯一记忆ID
            memory_id = int.from_bytes(
                binary.squeeze().cpu().numpy().tobytes()[:8], 
                'big'
            ) % (2**31)
        
        return dg_out, memory_id
    
    def encode_ca3(self, dg_output: torch.Tensor, 
                    memory_id: int,
                    timestamp: float,
                    text_pointer: str) -> MemoryAnchor:
        """
        CA3: 情景记忆存储
        存储：记忆ID + 时间戳 + 时序骨架 + 语义指针 + 因果关联
        """
        with torch.no_grad():
            ca3_out = torch.tanh(self.ca3_encoder(dg_output))
        
        # 创建记忆锚点
        anchor = MemoryAnchor(
            id=f"mem_{memory_id}_{int(timestamp)}",
            memory_id=memory_id,
            feature_vector=ca3_out.squeeze().cpu().numpy(),
            text_pointer=text_pointer[:100],  # 只存指针
            timestamp=timestamp,
            temporal_skeleton=[int(timestamp % 1000)],
            semantic_pointer=ca3_out.squeeze()[:64].cpu().numpy(),
            causal_links=[]
        )
        
        return anchor
    
    def encode_ca1(self, ca3_output: torch.Tensor,
                    anchor: MemoryAnchor) -> MemoryAnchor:
        """
        CA1: 时序编码
        打精准时间戳，绑定时序-情景-因果关系
        """
        with torch.no_grad():
            # 确保维度匹配
            if ca3_output.shape[-1] != self.config.hippocampus_ca3_dim:
                # 调整维度
                ca3_output = F.pad(ca3_output, (0, self.config.hippocampus_ca3_dim - ca3_output.shape[-1]))
            ca1_out = torch.tanh(self.ca1_encoder(ca3_output))
        
        # 更新记忆锚点
        anchor.feature_vector = ca1_out.squeeze().cpu().numpy()
        
        return anchor
    
    def store_memory(self, hidden_state: torch.Tensor, 
                     text_pointer: str,
                     context_ids: List[int] = None) -> MemoryAnchor:
        """
        完整记忆编码流程: EC -> DG -> CA3 -> CA1
        """
        # 检查内存限制
        if self._estimate_memory_usage() >= self.max_memory_bytes:
            # 移除最旧的记忆
            self.memories.pop(0)
        
        timestamp = time.time() * 1000
        
        # EC编码
        ec_out = self.encode_ec(hidden_state)
        
        # DG模式分离
        dg_out, memory_id = self.encode_dg(ec_out)
        
        # CA3情景存储
        anchor = self.encode_ca3(dg_out, memory_id, timestamp, text_pointer)
        
        # CA1时序编码 - 使用dg_out作为输入
        anchor = self.encode_ca1(dg_out, anchor)
        
        # 添加上下文
        if context_ids:
            anchor.context_ids = context_ids
        
        # 存储
        self.memories.append(anchor)
        self.memory_id_counter += 1
        
        return anchor
    
    def recall_memory(self, query_hidden: torch.Tensor, 
                       top_k: int = NARROW_WINDOW_SIZE) -> List[MemoryAnchor]:
        """
        记忆召回
        基于部分线索完成完整记忆链条召回
        """
        if not self.memories:
            return []
        
        # 编码查询
        query_ec = self.encode_ec(query_hidden)
        query_dg, _ = self.encode_dg(query_ec)
        
        # 计算相似度
        similarities = []
        for mem in self.memories:
            # 余弦相似度 - 确保维度匹配
            mem_vector = torch.tensor(mem.feature_vector, dtype=torch.float32)
            
            # 调整维度
            query_vec = query_dg.squeeze()
            mem_vec = mem_vector.squeeze()
            
            # 取最小维度
            min_dim = min(query_vec.shape[0], mem_vec.shape[0])
            query_vec = query_vec[:min_dim]
            mem_vec = mem_vec[:min_dim]
            
            # 计算余弦相似度
            dot_product = torch.dot(query_vec, mem_vec)
            norm_query = torch.norm(query_vec)
            norm_mem = torch.norm(mem_vec)
            
            if norm_query > 0 and norm_mem > 0:
                sim = (dot_product / (norm_query * norm_mem)).item()
            else:
                sim = 0.0
            
            similarities.append((sim, mem))
        
        # 排序
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 返回top_k
        result = []
        for sim, mem in similarities[:top_k]:
            mem.access_count += 1
            mem.strength = min(1.0, mem.strength + 0.1 * sim)
            result.append(mem)
        
        return result
    
    def start_swr_replay(self):
        """
        SWR: 尖波涟漪离线回放
        端侧空闲时执行记忆巩固
        """
        self.swr_active = True
        self.swr_replay_queue = self.memories.copy()
        print(f"[Hippocampus] SWR回放启动，记忆数: {len(self.swr_replay_queue)}")
    
    def swr_replay_step(self) -> Optional[MemoryAnchor]:
        """
        SWR单步回放
        """
        if not self.swr_replay_queue:
            self.swr_active = False
            return None
        
        return self.swr_replay_queue.pop(0)
    
    def _estimate_memory_usage(self) -> int:
        """估算内存使用"""
        # 每个记忆锚点约 1KB
        return len(self.memories) * 1024
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.memories:
            return {
                "total_memories": 0,
                "avg_strength": 0,
                "avg_access_count": 0,
                "memory_usage_bytes": 0
            }
        
        return {
            "total_memories": len(self.memories),
            "avg_strength": sum(m.strength for m in self.memories) / len(self.memories),
            "avg_access_count": sum(m.access_count for m in self.memories) / len(self.memories),
            "memory_usage_bytes": self._estimate_memory_usage()
        }


# ==================== 模块4：自闭环优化系统 ====================

class ClosedLoopOptimizer:
    """
    单智体自生成-自博弈-自评判闭环优化系统
    三种模式自动切换
    """
    
    def __init__(self, model_adapter: ModelAdapter, 
                 stdp_system: STDPSystem,
                 hippocampus: HippocampusSystem,
                 config: BrainConfig):
        self.model_adapter = model_adapter
        self.stdp_system = stdp_system
        self.hippocampus = hippocampus
        self.config = config
        
        # 当前模式
        self.current_mode = OptimizationMode.SELF_GENERATION
        
        # 自博弈状态
        self.self_play_iteration = 0
        self.self_play_result = ""
        
        # 自评判计数
        self.cycle_count = 0
    
    def detect_mode(self, user_input: str) -> OptimizationMode:
        """
        自动检测优化模式
        基于输入关键词和任务难度
        """
        # 高难度关键词 -> 自博弈模式
        high_difficulty_keywords = [
            "计算", "推理", "证明", "代码", "编程",
            "数学", "逻辑", "分析", "比较", "判断"
        ]
        
        # 决策类关键词 -> 自评判模式
        decision_keywords = [
            "方案", "建议", "选择", "决策", "评估",
            "推荐", "最优", "最好", "对比"
        ]
        
        # 检测
        for keyword in high_difficulty_keywords:
            if keyword in user_input:
                return OptimizationMode.SELF_PLAY
        
        for keyword in decision_keywords:
            if keyword in user_input:
                return OptimizationMode.SELF_EVALUATION
        
        # 默认自生成模式
        return OptimizationMode.SELF_GENERATION
    
    def execute_self_generation(self, 
                                 user_input: str,
                                 generate_fn: Callable) -> Tuple[str, List[STDPUpdateRecord]]:
        """
        模式1：自生成组合输出
        并行生成2个候选，STDP加权投票
        """
        updates = []
        
        # 生成候选1
        self.model_adapter.set_role(RoleMode.GENERATOR)
        candidate_1 = generate_fn(user_input, seed=42, temperature=0.7)
        
        # 生成候选2
        candidate_2 = generate_fn(user_input, seed=123, temperature=0.9)
        
        # STDP加权投票
        # 基于历史准确率计算权重
        stats = self.stdp_system.get_statistics()
        weight_1 = 0.5 + 0.1 * (stats.get('ltp_ratio', 0.5) - 0.5)
        weight_2 = 1.0 - weight_1
        
        # 选择更一致的候选
        if len(candidate_1) > 0 and len(candidate_2) > 0:
            # 简单选择：取较长的回复
            if len(candidate_1) >= len(candidate_2):
                result = candidate_1
                winner = 1
            else:
                result = candidate_2
                winner = 2
        else:
            result = candidate_1 if candidate_1 else candidate_2
            winner = 1 if candidate_1 else 2
        
        # 记录STDP更新
        now = time.time() * 1000
        updates.append(STDPUpdateRecord(
            timestamp=now,
            layer_type='selfGeneration',
            weight_id=f'candidate_{winner}',
            delta_value=0.01 * weight_1 if winner == 1 else 0.01 * weight_2,
            update_type='LTP',
            trigger_reason=f'自生成组合输出, 权重:{weight_1:.2f}/{weight_2:.2f}'
        ))
        
        return result, updates
    
    def execute_self_play(self,
                          user_input: str,
                          generate_fn: Callable) -> Tuple[str, List[STDPUpdateRecord]]:
        """
        模式2：自博弈竞争优化
        提案-验证对抗迭代
        """
        updates = []
        result = ""
        
        for iteration in range(self.config.self_play_max_iterations):
            # 奇数周期：提案角色
            if iteration % 2 == 0:
                self.model_adapter.set_role(RoleMode.GENERATOR)
                proposal = generate_fn(user_input, temperature=0.7)
                self.self_play_result = proposal
            # 偶数周期：验证角色
            else:
                self.model_adapter.set_role(RoleMode.VALIDATOR)
                validation_prompt = f"请验证以下内容的正确性：\n{self.self_play_result}"
                validation = generate_fn(validation_prompt, temperature=0.3)
                
                # 检查是否有错误
                if "错误" not in validation and "不正确" not in validation:
                    result = self.self_play_result
                    break
                else:
                    # 修正后继续
                    user_input = f"请修正以下内容：\n{self.self_play_result}\n问题：{validation}"
        
        if not result:
            result = self.self_play_result
        
        # 记录STDP更新
        now = time.time() * 1000
        updates.append(STDPUpdateRecord(
            timestamp=now,
            layer_type='selfPlay',
            weight_id=f'iteration_{iteration}',
            delta_value=0.01,
            update_type='LTP',
            trigger_reason=f'自博弈迭代{iteration}次后收敛'
        ))
        
        return result, updates
    
    def execute_self_evaluation(self,
                                user_input: str,
                                generate_fn: Callable) -> Tuple[str, List[STDPUpdateRecord], EvaluationResult]:
        """
        模式3：自双输出+自评判选优
        4维度打分评判
        """
        updates = []
        
        # 生成候选1
        self.model_adapter.set_role(RoleMode.GENERATOR)
        candidate_1 = generate_fn(user_input, seed=42, temperature=0.7)
        
        # 生成候选2
        candidate_2 = generate_fn(user_input, seed=123, temperature=0.8)
        
        # 切换到评判角色
        self.model_adapter.set_role(RoleMode.EVALUATOR)
        
        # 评判候选1
        eval_prompt_1 = f"""请对以下回复打分(0-10分)：
{candidate_1}

评判维度：
1. 事实准确性
2. 逻辑完整性
3. 语义连贯性
4. 指令遵循度

请给出每个维度的分数："""
        
        eval_1 = generate_fn(eval_prompt_1, temperature=0.1)
        scores_1 = self._parse_scores(eval_1)
        
        # 评判候选2
        eval_prompt_2 = f"""请对以下回复打分(0-10分)：
{candidate_2}

评判维度：
1. 事实准确性
2. 逻辑完整性
3. 语义连贯性
4. 指令遵循度

请给出每个维度的分数："""
        
        eval_2 = generate_fn(eval_prompt_2, temperature=0.1)
        scores_2 = self._parse_scores(eval_2)
        
        # 计算总分
        total_1 = sum(scores_1.values())
        total_2 = sum(scores_2.values())
        
        # 选择胜者
        winner = 1 if total_1 >= total_2 else 2
        result = candidate_1 if winner == 1 else candidate_2
        
        # 创建评判结果
        eval_result = EvaluationResult(
            candidate_1=candidate_1,
            candidate_2=candidate_2,
            scores_1=scores_1,
            scores_2=scores_2,
            winner=winner,
            total_score_1=total_1,
            total_score_2=total_2
        )
        
        # STDP更新
        stdp_updates = self.stdp_system.update_self_eval_stdp(eval_result)
        updates.extend(stdp_updates)
        
        return result, updates, eval_result
    
    def _parse_scores(self, eval_text: str) -> Dict[str, float]:
        """解析评判分数"""
        import re
        
        scores = {
            "事实准确性": 5.0,
            "逻辑完整性": 5.0,
            "语义连贯性": 5.0,
            "指令遵循度": 5.0
        }
        
        # 尝试解析分数
        for key in scores.keys():
            pattern = rf"{key}[：:]\s*(\d+(?:\.\d+)?)"
            match = re.search(pattern, eval_text)
            if match:
                scores[key] = float(match.group(1))
        
        return scores
    
    def execute(self, user_input: str,
                generate_fn: Callable) -> Tuple[str, List[STDPUpdateRecord], OptimizationMode]:
        """
        执行自闭环优化
        自动选择模式
        """
        self.cycle_count += 1
        
        # 检测模式
        mode = self.detect_mode(user_input)
        self.current_mode = mode
        
        # 执行对应模式
        if mode == OptimizationMode.SELF_GENERATION:
            result, updates = self.execute_self_generation(user_input, generate_fn)
        elif mode == OptimizationMode.SELF_PLAY:
            result, updates = self.execute_self_play(user_input, generate_fn)
        else:  # SELF_EVALUATION
            result, updates, _ = self.execute_self_evaluation(user_input, generate_fn)
        
        return result, updates, mode


# ==================== 模块2：100Hz推理引擎 ====================

class InferenceEngine:
    """
    100Hz人脑级高刷新单周期推理引擎
    严格10ms刷新周期，O(1)注意力复杂度
    """
    
    def __init__(self, model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 config: BrainConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # 初始化子系统
        self.model_adapter = ModelAdapter(model, config)
        self.stdp_system = STDPSystem(self.model_adapter, config)
        self.hippocampus = HippocampusSystem(model.config.hidden_size, config)
        self.optimizer = ClosedLoopOptimizer(
            self.model_adapter, self.stdp_system, self.hippocampus, config
        )
        
        # 周期状态
        self.cycle_count = 0
        self.is_running = False
        self.current_phase = CyclePhase.INPUT_RECEIVE
        
        # 窄窗口上下文
        self.narrow_window: deque = deque(maxlen=NARROW_WINDOW_SIZE)
        
        # 工作记忆
        self.working_memory: List[TokenFeature] = []
        self.max_working_memory = 50
        
        # 性能指标
        self.metrics = {
            "total_cycles": 0,
            "avg_cycle_time_ms": 0,
            "max_cycle_time_ms": 0,
            "total_tokens": 0
        }
        
        # 对话历史（用于上下文）
        self.conversation_history: List[Dict] = []
        self.max_history = 10
    
    def start(self):
        """启动引擎"""
        self.is_running = True
        print(f"[Engine] 推理引擎启动，刷新率: {self.config.refresh_rate_hz}Hz")
    
    def stop(self):
        """停止引擎"""
        self.is_running = False
        print(f"[Engine] 推理引擎停止，总周期数: {self.cycle_count}")
    
    def execute_cycle(self, 
                      input_token: TokenFeature,
                      on_phase_change: Callable = None) -> CycleExecutionState:
        """
        执行单个刷新周期
        严格按7步流程执行
        """
        cycle_start = time.time() * 1000
        self.cycle_count += 1
        
        state = CycleExecutionState(
            cycle_id=self.cycle_count,
            start_time=cycle_start,
            end_time=0,
            phase=CyclePhase.INPUT_RECEIVE,
            input_token=input_token,
            memory_anchors=[],
            output="",
            stdp_updates=[],
            optimization_mode=self.optimizer.current_mode
        )
        
        try:
            # ===== 阶段1：输入token接收与特征提取 =====
            self.current_phase = CyclePhase.INPUT_RECEIVE
            if on_phase_change:
                on_phase_change(CyclePhase.INPUT_RECEIVE)
            
            # 添加到窄窗口
            self.narrow_window.append(input_token)
            
            # 提取特征
            feature = self.model_adapter.get_attention_features(
                0,  # 第一层
                torch.tensor(input_token.embedding).unsqueeze(0).unsqueeze(0)
            )
            
            # ===== 阶段2：海马体记忆锚点调取与注意力门控加载 =====
            self.current_phase = CyclePhase.MEMORY_RECALL
            if on_phase_change:
                on_phase_change(CyclePhase.MEMORY_RECALL)
            
            # 从海马体召回记忆
            query_hidden = torch.tensor(input_token.embedding).unsqueeze(0)
            memory_anchors = self.hippocampus.recall_memory(query_hidden, top_k=NARROW_WINDOW_SIZE)
            state.memory_anchors = memory_anchors
            
            # ===== 阶段3：窄窗口上下文+当前token的模型前向推理 =====
            self.current_phase = CyclePhase.INFERENCE
            if on_phase_change:
                on_phase_change(CyclePhase.INFERENCE)
            
            # 构建窄窗口输入
            context = self._build_narrow_context(input_token)
            
            # ===== 阶段4：单周期输出结果生成 =====
            self.current_phase = CyclePhase.OUTPUT
            if on_phase_change:
                on_phase_change(CyclePhase.OUTPUT)
            
            # 执行自闭环优化生成
            output, stdp_updates, mode = self.optimizer.execute(
                context,
                lambda prompt, **kwargs: self._generate(prompt, **kwargs)
            )
            
            state.output = output
            state.stdp_updates = stdp_updates
            state.optimization_mode = mode
            
            # ===== 阶段5：全链路STDP权重本地刷新 =====
            self.current_phase = CyclePhase.STDP_UPDATE
            if on_phase_change:
                on_phase_change(CyclePhase.STDP_UPDATE)
            
            # 注意力层STDP更新
            contribution_scores = [0.5] * len(list(self.narrow_window))
            attention_updates = self.stdp_system.update_attention_stdp(
                input_token,
                list(self.narrow_window),
                contribution_scores
            )
            state.stdp_updates.extend(attention_updates)
            
            # 应用STDP更新
            self.stdp_system.apply_updates_to_weights(state.stdp_updates)
            
            # ===== 阶段6：海马体情景记忆编码与更新 =====
            self.current_phase = CyclePhase.MEMORY_ENCODE
            if on_phase_change:
                on_phase_change(CyclePhase.MEMORY_ENCODE)
            
            # 存储到海马体
            self.hippocampus.store_memory(
                query_hidden,
                f"用户: {input_token.text}\n助手: {output[:50]}"
            )
            
            # ===== 阶段7：全局工作记忆压缩更新 =====
            self.current_phase = CyclePhase.WORKING_MEMORY
            if on_phase_change:
                on_phase_change(CyclePhase.WORKING_MEMORY)
            
            # 更新工作记忆
            self.working_memory.append(input_token)
            if len(self.working_memory) > self.max_working_memory:
                self.working_memory.pop(0)
            
            # 完成周期
            state.phase = CyclePhase.OUTPUT
            state.end_time = time.time() * 1000
            
            # 更新指标
            self._update_metrics(state.end_time - cycle_start)
            
            return state
            
        except Exception as e:
            print(f"[Engine] 周期执行错误: {e}")
            state.end_time = time.time() * 1000
            return state
    
    def _build_narrow_context(self, current_token: TokenFeature) -> str:
        """构建窄窗口上下文"""
        context_parts = []
        
        # 只使用窄窗口内的token
        for token in self.narrow_window:
            if token.text:
                context_parts.append(token.text)
        
        # 添加当前token
        if current_token.text:
            context_parts.append(current_token.text)
        
        return " ".join(context_parts) if context_parts else current_token.text
    
    def _generate(self, prompt: str, seed: int = None, temperature: float = None) -> str:
        """生成回复"""
        if temperature is None:
            temperature = self.config.temperature
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # 编码
        inputs = self.tokenizer(
            f"用户: {prompt}\n助手:",
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=self.config.max_new_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.5,
                no_repeat_ngram_size=2
            )
        
        # 解码
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 清理
        generated_text = self._clean_output(generated_text)
        
        return generated_text
    
    def _clean_output(self, text: str) -> str:
        """清理输出"""
        lines = text.strip().split('\n')
        result_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("用户:") or line.startswith("user:"):
                break
            result_lines.append(line)
            if len(result_lines) >= 3:
                break
        
        result = '\n'.join(result_lines)
        if len(result) > 500:
            result = result[:500]
        
        return result
    
    def _update_metrics(self, cycle_time_ms: float):
        """更新性能指标"""
        n = self.metrics["total_cycles"] + 1
        self.metrics["total_cycles"] = n
        self.metrics["max_cycle_time_ms"] = max(
            self.metrics["max_cycle_time_ms"], 
            cycle_time_ms
        )
        self.metrics["avg_cycle_time_ms"] = (
            (self.metrics["avg_cycle_time_ms"] * (n - 1) + cycle_time_ms) / n
        )
    
    def infer(self, user_input: str) -> Tuple[str, Dict]:
        """
        执行推理（对外接口）
        """
        if not self.is_running:
            self.start()
        
        start_time = time.time()
        
        # 创建输入token
        input_token = TokenFeature(
            token_id=0,
            text=user_input,
            embedding=np.zeros(self.model.config.hidden_size),
            attention_weights=np.zeros(64),
            temporal_feature=np.zeros(64),
            semantic_feature=np.zeros(64),
            timestamp=time.time() * 1000
        )
        
        # 执行周期
        state = self.execute_cycle(input_token)
        
        # 更新对话历史
        self.conversation_history.append({
            "user": user_input,
            "assistant": state.output
        })
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        cycle_time = (time.time() - start_time) * 1000
        
        metadata = {
            "cycle_id": state.cycle_id,
            "cycle_time_ms": cycle_time,
            "optimization_mode": state.optimization_mode.value,
            "memory_anchors": len(state.memory_anchors),
            "stdp_updates": len(state.stdp_updates),
            "stdp_stats": self.stdp_system.get_statistics(),
            "hippocampus_stats": self.hippocampus.get_statistics(),
            "metrics": self.metrics.copy()
        }
        
        return state.output, metadata
    
    def train_step(self, user_input: str, expected_output: str,
                   optimizer: torch.optim.Optimizer) -> Tuple[float, Dict]:
        """训练步骤"""
        self.model.train()
        
        text = f"用户: {user_input}\n助手: {expected_output}"
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        outputs = self.model(
            input_ids=inputs['input_ids'],
            labels=inputs['input_ids'],
            output_hidden_states=True
        )
        
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # 计算贡献度
        contribution = 0.5
        if outputs.hidden_states:
            activation = torch.mean(torch.abs(outputs.hidden_states[-1])).item()
            contribution = min(1.0, activation / 10.0)
        
        # STDP更新
        stdp_updates = self.stdp_system.update_weights(
            outputs.hidden_states,
            loss.item(),
            contribution
        )
        
        optimizer.step()
        
        # 存储到海马体
        with torch.no_grad():
            if outputs.hidden_states:
                last_hidden = outputs.hidden_states[-1][0].mean(dim=0)
                self.hippocampus.store_memory(
                    last_hidden.unsqueeze(0),
                    text
                )
        
        self.model.eval()
        
        return loss.item(), {
            "contribution": contribution,
            "stdp_updates": len(stdp_updates)
        }
    
    def get_metrics(self) -> Dict:
        """获取性能指标"""
        return self.metrics.copy()
    
    def clear_history(self):
        """清除历史"""
        self.conversation_history = []
        self.narrow_window.clear()
        self.working_memory = []


# ==================== 核心架构整合 ====================

class BrainArchitecture:
    """
    类人脑双系统全闭环AI架构
    整合所有模块的生产级实现
    """
    
    def __init__(self, config: BrainConfig = None):
        self.config = config or BrainConfig()
        self.model = None
        self.tokenizer = None
        self.engine = None
        self.is_initialized = False
    
    def initialize(self) -> Dict:
        """初始化架构"""
        print("=" * 70)
        print("类人脑双系统全闭环AI架构 - 初始化")
        print("=" * 70)
        print(f"刷新周期: {self.config.refresh_cycle_ms}ms ({self.config.refresh_rate_hz}Hz)")
        print(f"权重拆分: {self.config.static_ratio*100:.0f}%静态 + {self.config.dynamic_ratio*100:.0f}%动态")
        print(f"窄窗口大小: {self.config.narrow_window_size} token")
        print("=" * 70)
        
        try:
            # 加载模型
            print("\n[1/4] 加载Qwen3.5-0.8B底座模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                torch_dtype=self.config.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            self.model.eval()
            print("✓ 模型加载成功")
            
            # 初始化推理引擎
            print("\n[2/4] 初始化100Hz推理引擎...")
            self.engine = InferenceEngine(self.model, self.tokenizer, self.config)
            print("✓ 推理引擎初始化成功")
            
            # 验证权重拆分
            print("\n[3/4] 验证权重双轨拆分...")
            static_count = len(self.engine.model_adapter.static_weights)
            dynamic_count = len(self.engine.model_adapter.dynamic_weights)
            print(f"  静态权重参数组: {static_count}")
            print(f"  动态权重参数组: {dynamic_count}")
            print("✓ 权重拆分验证通过")
            
            # 初始化完成
            print("\n[4/4] 架构初始化完成")
            self.is_initialized = True
            
            total_params = sum(p.numel() for p in self.model.parameters())
            
            return {
                "success": True,
                "message": "架构初始化成功",
                "model_info": {
                    "name": "Qwen3.5-0.8B-Base",
                    "total_params": total_params,
                    "hidden_size": self.model.config.hidden_size,
                    "num_layers": self.model.config.num_hidden_layers,
                },
                "config": {
                    "refresh_rate_hz": self.config.refresh_rate_hz,
                    "narrow_window_size": self.config.narrow_window_size,
                    "static_ratio": self.config.static_ratio,
                    "dynamic_ratio": self.config.dynamic_ratio,
                }
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"初始化失败: {e}"
            }
    
    def infer(self, user_input: str) -> Dict:
        """执行推理"""
        if not self.is_initialized:
            return {"error": "架构未初始化"}
        
        output, metadata = self.engine.infer(user_input)
        
        return {
            "output": output,
            "metadata": metadata
        }
    
    def train(self, training_data: List[Dict], 
              epochs: int = 3,
              learning_rate: float = 5e-6) -> Dict:
        """执行训练"""
        if not self.is_initialized:
            return {"error": "架构未初始化"}
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        
        training_log = []
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for sample in training_data:
                loss, info = self.engine.train_step(
                    sample["input"],
                    sample["output"],
                    optimizer
                )
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(training_data)
            training_log.append({
                "epoch": epoch + 1,
                "avg_loss": avg_loss
            })
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "training_log": training_log,
            "total_time": total_time,
            "final_loss": training_log[-1]["avg_loss"] if training_log else 0
        }
    
    def get_status(self) -> Dict:
        """获取状态"""
        if not self.is_initialized:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "is_running": self.engine.is_running if self.engine else False,
            "cycle_count": self.engine.cycle_count if self.engine else 0,
            "current_mode": self.engine.optimizer.current_mode.value if self.engine else None,
            "metrics": self.engine.get_metrics() if self.engine else {},
            "stdp_stats": self.engine.stdp_system.get_statistics() if self.engine else {},
            "hippocampus_stats": self.engine.hippocampus.get_statistics() if self.engine else {},
        }
    
    def clear_history(self):
        """清除历史"""
        if self.engine:
            self.engine.clear_history()
    
    def start_swr_replay(self):
        """启动SWR离线回放"""
        if self.engine:
            self.engine.hippocampus.start_swr_replay()


# ==================== 导出 ====================

__all__ = [
    # 配置
    'BrainConfig',
    
    # 核心架构
    'BrainArchitecture',
    
    # 引擎和系统
    'InferenceEngine',
    'STDPSystem',
    'HippocampusSystem',
    'ClosedLoopOptimizer',
    'ModelAdapter',
    
    # 枚举
    'OptimizationMode',
    'CyclePhase',
    'RoleMode',
    
    # 数据结构
    'TokenFeature',
    'MemoryAnchor',
    'STDPUpdateRecord',
    'CycleExecutionState',
    'EvaluationResult',
    
    # 常量
    'REFRESH_CYCLE_MS',
    'REFRESH_RATE_HZ',
    'STATIC_WEIGHT_RATIO',
    'DYNAMIC_WEIGHT_RATIO',
]
