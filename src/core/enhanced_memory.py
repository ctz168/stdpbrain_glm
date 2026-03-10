#!/usr/bin/env python3
"""
增强版记忆系统
Enhanced Memory System

改进：
1. 工作记忆 - 短期记忆保持
2. 海马体增强 - 更好的编码和召回
3. 记忆整合 - 工作记忆与海马体协同
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import re

# ==================== 数据结构 ====================

@dataclass
class MemoryItem:
    """记忆项"""
    id: str
    content: str
    memory_type: str  # 'working', 'short_term', 'long_term'
    embedding: np.ndarray = None
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    strength: float = 1.0
    context: List[str] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)  # 关键实体
    importance: float = 0.5  # 重要性评分


# ==================== 工作记忆系统 ====================

class WorkingMemory:
    """
    工作记忆系统
    模拟前额叶皮层的工作记忆功能
    保持当前对话的上下文信息
    """
    
    def __init__(self, capacity: int = 7):
        """
        容量默认为7（符合人类工作记忆容量）
        """
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)
        self.attention_weights: List[float] = []
        
    def add(self, content: str, key_entities: List[str] = None):
        """添加到工作记忆"""
        item = MemoryItem(
            id=f"wm_{len(self.items)}_{time.time()}",
            content=content,
            memory_type='working',
            key_entities=key_entities or [],
            importance=self._calculate_importance(content)
        )
        self.items.append(item)
        self._update_attention()
        
    def _calculate_importance(self, content: str) -> float:
        """计算内容重要性"""
        importance = 0.5
        
        # 包含数字
        if re.search(r'\d+', content):
            importance += 0.1
        
        # 包含关键实体标记
        keywords = ['记住', '叫', '是', '喜欢', '生日', '住', '电话', '名字']
        for kw in keywords:
            if kw in content:
                importance += 0.1
        
        # 是问题
        if '？' in content or '?' in content:
            importance += 0.1
            
        return min(1.0, importance)
    
    def _update_attention(self):
        """更新注意力权重"""
        if not self.items:
            self.attention_weights = []
            return
        
        # 最近的项目权重更高
        n = len(self.items)
        self.attention_weights = [(i + 1) / n for i in range(n)]
    
    def get_context(self) -> str:
        """获取工作记忆上下文"""
        if not self.items:
            return ""
        
        context_parts = []
        for item in self.items:
            context_parts.append(item.content)
        
        return "\n".join(context_parts)
    
    def find_entity(self, entity_type: str) -> Optional[str]:
        """查找特定类型的实体"""
        patterns = {
            'name': [r'我叫(\w+)', r'名字是(\w+)', r'我是(\w+)'],
            'birthday': [r'生日是(\d+月\d+日?)', r'(\d+月\d+日?)生日'],
            'address': [r'住在(.+)', r'地址是(.+)', r'住在(.+)'],
            'phone': [r'电话是(\d+)', r'手机是(\d+)', r'号码是(\d+)'],
            'color': [r'喜欢(.+?)色', r'颜色是(.+)', r'喜欢(.+)'],
            'number': [r'记住.*?(\d+)', r'数字是(\d+)', r'(\d+)'],
        }
        
        if entity_type not in patterns:
            return None
        
        # 从最近的记忆开始搜索
        for item in reversed(self.items):
            for pattern in patterns[entity_type]:
                match = re.search(pattern, item.content)
                if match:
                    return match.group(1)
        
        return None
    
    def clear(self):
        """清空工作记忆"""
        self.items.clear()
        self.attention_weights = []
    
    def get_statistics(self) -> Dict:
        return {
            "capacity": self.capacity,
            "current_items": len(self.items),
            "utilization": len(self.items) / self.capacity
        }


# ==================== 增强版海马体系统 ====================

class EnhancedHippocampus:
    """
    增强版海马体系统
    实现更强的记忆编码和召回
    """
    
    def __init__(self, embedding_dim: int = 256, memory_size: int = 100):
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        
        # 记忆存储
        self.memories: List[MemoryItem] = []
        
        # 索引结构（用于快速检索）
        self.entity_index: Dict[str, List[int]] = {}  # 实体 -> 记忆索引
        
        # 编码网络
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Tanh()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取文本中的关键实体"""
        entities = []
        
        # 提取数字
        numbers = re.findall(r'\d+', text)
        entities.extend(numbers)
        
        # 提取日期
        dates = re.findall(r'\d+月\d+日?', text)
        entities.extend(dates)
        
        # 提取人名（简单规则）
        names = re.findall(r'我叫(\w+)|名字是(\w+)|我是(\w+)', text)
        for name_tuple in names:
            for name in name_tuple:
                if name:
                    entities.append(name)
        
        # 提取地点
        addresses = re.findall(r'住在(.+?)(?:，|。|$)', text)
        entities.extend(addresses)
        
        # 提取颜色
        colors = re.findall(r'喜欢(.+?)色', text)
        entities.extend(colors)
        
        return entities
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """创建文本嵌入"""
        # 简单的嵌入：使用字符级别的特征
        embedding = np.zeros(self.embedding_dim)
        
        # 字符频率
        for i, char in enumerate(text[:self.embedding_dim]):
            embedding[i % self.embedding_dim] += ord(char) / 65536.0
        
        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def store(self, text: str, context: List[str] = None) -> MemoryItem:
        """存储记忆"""
        if len(self.memories) >= self.memory_size:
            # 移除最弱且最少访问的记忆
            self.memories.sort(key=lambda m: m.strength * m.access_count)
            removed = self.memories.pop(0)
            # 更新索引
            for entity in removed.key_entities:
                if entity in self.entity_index:
                    self.entity_index[entity] = [i for i in self.entity_index[entity] if i != 0]
        
        # 提取实体
        entities = self._extract_entities(text)
        
        # 创建嵌入
        embedding = self._create_embedding(text)
        
        # 计算重要性
        importance = 0.5
        if entities:
            importance += 0.2
        if '记住' in text or '叫' in text:
            importance += 0.2
        
        # 创建记忆项
        memory = MemoryItem(
            id=f"hip_{len(self.memories)}_{time.time()}",
            content=text,
            memory_type='long_term',
            embedding=embedding,
            context=context or [],
            key_entities=entities,
            importance=min(1.0, importance)
        )
        
        # 添加到存储
        idx = len(self.memories)
        self.memories.append(memory)
        
        # 更新索引
        for entity in entities:
            if entity not in self.entity_index:
                self.entity_index[entity] = []
            self.entity_index[entity].append(idx)
        
        return memory
    
    def recall_by_entity(self, entity: str) -> List[MemoryItem]:
        """通过实体召回记忆"""
        if entity not in self.entity_index:
            return []
        
        results = []
        for idx in self.entity_index[entity]:
            if idx < len(self.memories):
                memory = self.memories[idx]
                memory.access_count += 1
                memory.strength = min(1.0, memory.strength + 0.1)
                results.append(memory)
        
        return results
    
    def recall_by_query(self, query: str, top_k: int = 3) -> List[MemoryItem]:
        """通过查询召回记忆"""
        if not self.memories:
            return []
        
        # 提取查询中的实体
        query_entities = self._extract_entities(query)
        
        # 首先尝试实体匹配
        entity_results = []
        for entity in query_entities:
            entity_results.extend(self.recall_by_entity(entity))
        
        if entity_results:
            return entity_results[:top_k]
        
        # 如果没有实体匹配，使用相似度搜索
        query_embedding = self._create_embedding(query)
        
        similarities = []
        for i, memory in enumerate(self.memories):
            if memory.embedding is not None:
                sim = np.dot(query_embedding, memory.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-8
                )
                # 考虑记忆强度和访问频率
                adjusted_sim = sim * (0.5 + 0.5 * memory.strength) * (1 + 0.1 * memory.access_count)
                similarities.append((adjusted_sim, i, memory))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for sim, idx, memory in similarities[:top_k]:
            memory.access_count += 1
            memory.strength = min(1.0, memory.strength + 0.1 * sim)
            results.append(memory)
        
        return results
    
    def get_all_entities(self) -> Dict[str, List[str]]:
        """获取所有实体及其对应的记忆内容"""
        result = {}
        for entity, indices in self.entity_index.items():
            result[entity] = [self.memories[idx].content for idx in indices if idx < len(self.memories)]
        return result
    
    def get_statistics(self) -> Dict:
        if not self.memories:
            return {
                "total_memories": 0,
                "entity_count": 0,
                "avg_strength": 0,
                "avg_access_count": 0
            }
        
        return {
            "total_memories": len(self.memories),
            "entity_count": len(self.entity_index),
            "avg_strength": sum(m.strength for m in self.memories) / len(self.memories),
            "avg_access_count": sum(m.access_count for m in self.memories) / len(self.memories),
            "entities": list(self.entity_index.keys())[:10]  # 前10个实体
        }


# ==================== 整合记忆系统 ====================

class IntegratedMemorySystem:
    """
    整合记忆系统
    结合工作记忆和海马体
    """
    
    def __init__(self, working_memory_capacity: int = 7, 
                 hippocampus_size: int = 100,
                 embedding_dim: int = 256):
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.hippocampus = EnhancedHippocampus(
            embedding_dim=embedding_dim,
            memory_size=hippocampus_size
        )
        
        # 记忆转移阈值
        self.consolidation_threshold = 0.6
    
    def process_input(self, user_input: str):
        """处理输入，更新记忆"""
        # 添加到工作记忆
        entities = self.hippocampus._extract_entities(user_input)
        self.working_memory.add(user_input, entities)
        
        # 如果包含重要信息，存储到海马体
        importance = self.working_memory._calculate_importance(user_input)
        if importance >= self.consolidation_threshold:
            self.hippocampus.store(user_input)
    
    def process_output(self, output: str):
        """处理输出，更新记忆"""
        self.working_memory.add(f"助手: {output}")
    
    def get_relevant_context(self, query: str) -> str:
        """获取相关上下文 - 简洁版本，不添加标记"""
        context_parts = []
        
        # 1. 从工作记忆获取最近的对话
        working_context = self.working_memory.get_context()
        if working_context:
            # 只保留最近的内容，不添加标记
            context_parts.append(working_context)
        
        # 2. 从海马体召回相关记忆
        hippocampus_memories = self.hippocampus.recall_by_query(query, top_k=2)
        if hippocampus_memories:
            memory_texts = [m.content for m in hippocampus_memories[:2]]
            context_parts.extend(memory_texts)
        
        # 返回简洁的上下文，不包含任何标记
        return "\n".join(context_parts) if context_parts else ""
    
    def find_in_memory(self, query: str) -> Optional[str]:
        """在记忆中查找特定信息"""
        # 提取查询类型
        query_type = self._detect_query_type(query)
        
        if query_type:
            # 先查工作记忆
            result = self.working_memory.find_entity(query_type)
            if result:
                return result
            
            # 再查海马体
            memories = self.hippocampus.recall_by_query(query, top_k=1)
            if memories:
                # 尝试从记忆中提取答案
                for entity in memories[0].key_entities:
                    return entity
        
        return None
    
    def _detect_query_type(self, query: str) -> Optional[str]:
        """检测查询类型"""
        patterns = {
            'name': [r'我叫什么', r'我的名字', r'我是谁'],
            'birthday': [r'我的生日', r'生日是什么'],
            'address': [r'我住哪', r'我的地址', r'住在哪里'],
            'phone': [r'我的电话', r'手机号'],
            'color': [r'我喜欢什么颜色', r'我的颜色'],
            'number': [r'记住的数字', r'那个数字', r'数字是多少'],
        }
        
        for qtype, pats in patterns.items():
            for pat in pats:
                if re.search(pat, query):
                    return qtype
        
        return None
    
    def get_statistics(self) -> Dict:
        return {
            "working_memory": self.working_memory.get_statistics(),
            "hippocampus": self.hippocampus.get_statistics()
        }


# ==================== 导出 ====================

__all__ = [
    'MemoryItem',
    'WorkingMemory',
    'EnhancedHippocampus',
    'IntegratedMemorySystem',
]
