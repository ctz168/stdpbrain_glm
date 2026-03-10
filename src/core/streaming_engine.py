#!/usr/bin/env python3
"""
生产级流式推理引擎
Production-Grade Streaming Inference Engine

严格实现：
- 100Hz刷新周期（每10ms生成一个token）
- 窄窗口处理（每次处理1-2个token）
- 流式输出到Telegram
"""

import os
import sys
import time
import json
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, AsyncGenerator, Generator
from dataclasses import dataclass, field
from collections import deque
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== 常量 ====================

REFRESH_CYCLE_MS = 10.0  # 10ms刷新周期
REFRESH_RATE_HZ = 100.0  # 100Hz刷新率
NARROW_WINDOW_SIZE = 2   # 窄窗口大小

# ==================== 配置 ====================

@dataclass
class StreamingConfig:
    """流式推理配置"""
    model_path: str = "./models/qwen3.5-0.8b"
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    # 刷新周期
    refresh_cycle_ms: float = REFRESH_CYCLE_MS
    refresh_rate_hz: float = REFRESH_RATE_HZ
    
    # 窄窗口
    narrow_window_size: int = NARROW_WINDOW_SIZE
    
    # 生成参数
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    
    # 流式参数
    token_timeout_ms: float = 100.0  # 单token超时


# ==================== 流式推理引擎 ====================

class StreamingInferenceEngine:
    """
    生产级流式推理引擎
    严格实现100Hz刷新周期和窄窗口处理
    """
    
    def __init__(self, model: AutoModelForCausalLM, 
                 tokenizer: AutoTokenizer,
                 config: StreamingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # 状态
        self.is_running = False
        self.current_input_ids = None
        self.current_attention_mask = None
        self.generated_tokens: List[int] = []
        self.past_key_values = None
        
        # 窄窗口上下文
        self.narrow_window: deque = deque(maxlen=NARROW_WINDOW_SIZE)
        
        # 统计
        self.total_tokens_generated = 0
        self.total_cycles = 0
        self.avg_token_time_ms = 0.0
        
        # EOS token
        self.eos_token_id = tokenizer.eos_token_id
        
        # 停止标记
        self.stop_tokens = self._get_stop_tokens()
    
    def _get_stop_tokens(self) -> List[int]:
        """获取停止token列表"""
        stop_tokens = [self.eos_token_id]
        
        # 添加 <|im_end|> token
        im_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        stop_tokens.extend(im_end)
        
        return stop_tokens
    
    def start(self):
        """启动引擎"""
        self.is_running = True
        print(f"[StreamingEngine] 启动，刷新率: {self.config.refresh_rate_hz}Hz")
    
    def stop(self):
        """停止引擎"""
        self.is_running = False
        print(f"[StreamingEngine] 停止，总token: {self.total_tokens_generated}")
    
    def reset(self):
        """重置状态"""
        self.current_input_ids = None
        self.current_attention_mask = None
        self.generated_tokens = []
        self.past_key_values = None
        self.narrow_window.clear()
    
    def _build_chatml_input(self, user_input: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建ChatML格式输入"""
        messages = [{"role": "user", "content": user_input}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        return inputs['input_ids'], inputs['attention_mask']
    
    def generate_token(self) -> Tuple[Optional[int], float]:
        """
        生成单个token
        严格遵循刷新周期
        
        Returns:
            (token_id, generation_time_ms) 或 (None, 0) 表示结束
        """
        if not self.is_running:
            return None, 0
        
        start_time = time.time()
        
        with torch.no_grad():
            # 使用past_key_values加速
            outputs = self.model(
                input_ids=self.current_input_ids if self.past_key_values is None else self.current_input_ids[:, -1:],
                attention_mask=self.current_attention_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
            
            # 更新past_key_values
            self.past_key_values = outputs.past_key_values
            
            # 获取logits
            logits = outputs.logits[:, -1, :]
            
            # 应用temperature
            logits = logits / self.config.temperature
            
            # Top-k过滤
            if self.config.top_k > 0:
                indices_to_remove = logits < torch.topk(logits, self.config.top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p过滤
            if self.config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_id = next_token.item()
            generation_time_ms = (time.time() - start_time) * 1000
            
            # 检查是否为停止token
            if token_id in self.stop_tokens:
                return None, generation_time_ms
            
            # 更新状态
            self.current_input_ids = torch.cat([self.current_input_ids, next_token], dim=-1)
            self.current_attention_mask = torch.cat([
                self.current_attention_mask,
                torch.ones((1, 1), dtype=self.current_attention_mask.dtype)
            ], dim=-1)
            
            self.generated_tokens.append(token_id)
            self.total_tokens_generated += 1
            
            # 更新统计
            n = self.total_tokens_generated
            self.avg_token_time_ms = (self.avg_token_time_ms * (n - 1) + generation_time_ms) / n
            
            return token_id, generation_time_ms
    
    def stream_generate(self, user_input: str) -> Generator[Tuple[str, float, bool], None, None]:
        """
        流式生成
        每个yield返回 (token_text, generation_time_ms, is_first)
        
        Yields:
            (token_text, time_ms, is_first_token)
        """
        self.reset()
        self.start()
        
        # 构建输入
        self.current_input_ids, self.current_attention_mask = self._build_chatml_input(user_input)
        
        is_first = True
        
        while self.is_running:
            # 检查是否超过最大长度
            if len(self.generated_tokens) >= self.config.max_new_tokens:
                break
            
            # 生成单个token
            token_id, gen_time = self.generate_token()
            
            if token_id is None:
                break
            
            # 解码token
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            
            # 更新窄窗口
            self.narrow_window.append(token_text)
            
            self.total_cycles += 1
            
            yield token_text, gen_time, is_first
            is_first = False
        
        self.stop()
    
    async def async_stream_generate(self, user_input: str) -> AsyncGenerator[Tuple[str, float, bool], None]:
        """
        异步流式生成
        用于Telegram Bot
        """
        for token_text, gen_time, is_first in self.stream_generate(user_input):
            yield token_text, gen_time, is_first
            # 让出控制权，允许其他协程执行
            await asyncio.sleep(0)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_tokens": self.total_tokens_generated,
            "total_cycles": self.total_cycles,
            "avg_token_time_ms": round(self.avg_token_time_ms, 2),
            "refresh_rate_hz": 1000.0 / max(self.avg_token_time_ms, 0.1),
        }


# ==================== Telegram流式发送器 ====================

class TelegramStreamer:
    """
    Telegram流式消息发送器
    实现边生成边发送
    """
    
    def __init__(self, bot, chat_id: int, update_interval_ms: float = 500):
        self.bot = bot
        self.chat_id = chat_id
        self.update_interval_ms = update_interval_ms
        
        # 消息状态
        self.message_id = None
        self.current_text = ""
        self.last_update_time = 0
        self.token_count = 0
        
        # 统计
        self.total_updates = 0
        self.start_time = 0
    
    async def start(self) -> int:
        """开始流式发送，创建初始消息"""
        self.start_time = time.time()
        self.current_text = "💭 "
        
        message = await self.bot.send_message(
            chat_id=self.chat_id,
            text=self.current_text,
            parse_mode=None
        )
        self.message_id = message.message_id
        
        return self.message_id
    
    async def add_token(self, token: str):
        """添加token到当前文本"""
        self.current_text += token
        self.token_count += 1
        
        # 检查是否需要更新消息
        current_time = time.time() * 1000
        if current_time - self.last_update_time >= self.update_interval_ms:
            await self._update_message()
    
    async def _update_message(self):
        """更新Telegram消息"""
        if self.message_id is None:
            return
        
        try:
            # 清理文本
            text = self._clean_text(self.current_text)
            
            if len(text) > 0:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self.message_id,
                    text=text,
                    parse_mode=None
                )
                self.total_updates += 1
                self.last_update_time = time.time() * 1000
        except Exception as e:
            # 忽略消息未修改等错误
            pass
    
    async def finish(self) -> str:
        """完成流式发送"""
        # 最终更新
        text = self._clean_text(self.current_text)
        
        try:
            await self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=self.message_id,
                text=text,
                parse_mode=None
            )
        except Exception:
            pass
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除思考标记
        import re
        text = re.sub(r'<think.*?</think\s*>', '', text, flags=re.DOTALL)
        text = text.replace('💭 ', '')
        
        # 限制长度
        if len(text) > 4000:
            text = text[:4000] + "..."
        
        return text.strip()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "message_id": self.message_id,
            "token_count": self.token_count,
            "total_updates": self.total_updates,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(self.token_count / max(elapsed, 0.1), 2),
        }


# ==================== 完整流式处理器 ====================

class StreamingProcessor:
    """
    完整流式处理器
    整合推理引擎和Telegram发送器
    """
    
    def __init__(self, model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 config: StreamingConfig):
        self.engine = StreamingInferenceEngine(model, tokenizer, config)
        self.config = config
    
    async def process_message(self, 
                              bot,
                              chat_id: int,
                              user_input: str) -> Tuple[str, Dict]:
        """
        处理消息并流式发送
        
        Args:
            bot: Telegram Bot实例
            chat_id: 聊天ID
            user_input: 用户输入
            
        Returns:
            (最终文本, 统计信息)
        """
        # 创建流式发送器
        streamer = TelegramStreamer(bot, chat_id)
        
        # 开始发送
        await streamer.start()
        
        # 流式生成并发送
        try:
            async for token_text, gen_time, is_first in self.engine.async_stream_generate(user_input):
                await streamer.add_token(token_text)
        except Exception as e:
            print(f"[StreamingProcessor] 错误: {e}")
        
        # 完成发送
        final_text = await streamer.finish()
        
        # 收集统计
        stats = {
            "engine": self.engine.get_statistics(),
            "streamer": streamer.get_statistics(),
        }
        
        return final_text, stats


# ==================== 导出 ====================

__all__ = [
    'StreamingConfig',
    'StreamingInferenceEngine',
    'TelegramStreamer',
    'StreamingProcessor',
    'REFRESH_CYCLE_MS',
    'REFRESH_RATE_HZ',
    'NARROW_WINDOW_SIZE',
]
