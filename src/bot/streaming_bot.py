#!/usr/bin/env python3
"""
生产级Telegram Bot - 流式输出
Production Telegram Bot with Streaming Output
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

import torch

# 配置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 尝试导入telegram
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    from telegram.request import HTTPXRequest
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.error("请安装: pip install python-telegram-bot")

# 导入核心模块
from core.streaming_engine import (
    StreamingConfig, StreamingProcessor,
    REFRESH_RATE_HZ, REFRESH_CYCLE_MS
)
from core.brain_architecture import BrainArchitecture, BrainConfig

# ==================== 全局变量 ====================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8533918353:AAG6Pxr0A4C4kJpCVjYzbtwtFzN4KZCcRag")

# 模型和处理器
model = None
tokenizer = None
streaming_processor = None
brain = None

# ==================== 初始化 ====================

def initialize():
    """初始化模型和处理器"""
    global model, tokenizer, streaming_processor, brain
    
    logger.info("初始化类人脑双系统AI架构...")
    print("=" * 60)
    print("类人脑双系统AI架构 - 生产级流式输出")
    print("=" * 60)
    print(f"刷新周期: {REFRESH_CYCLE_MS}ms ({REFRESH_RATE_HZ}Hz)")
    print("=" * 60)
    
    try:
        # 加载模型
        print("\n[1/3] 加载模型...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = "./models/qwen3.5-0.8b"
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        model.eval()
        print("✓ 模型加载成功")
        
        # 初始化流式处理器
        print("\n[2/3] 初始化流式处理器...")
        config = StreamingConfig()
        streaming_processor = StreamingProcessor(model, tokenizer, config)
        print("✓ 流式处理器初始化成功")
        
        # 初始化完整架构（用于训练等）
        print("\n[3/3] 初始化完整架构...")
        brain_config = BrainConfig()
        brain = BrainArchitecture(brain_config)
        brain.model = model
        brain.tokenizer = tokenizer
        brain.is_initialized = True
        print("✓ 完整架构初始化成功")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型参数: {total_params/1e6:.1f}M")
        
        return True
        
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 命令处理 ====================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /start 命令"""
    welcome = """
🧠 *类人脑双系统AI架构*

生产级流式输出系统

*核心特性:*
• 100Hz刷新周期 (每10ms生成一个token)
• 窄窗口处理 (1-2 token)
• 流式输出 (边生成边发送)
• STDP在线学习
• 海马体记忆系统

*命令:*
/start - 显示帮助
/clear - 清除对话
/status - 查看状态
/train - 开始训练

直接发送消息开始对话！
"""
    await update.message.reply_text(welcome, parse_mode='Markdown')


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /clear 命令"""
    if brain:
        brain.clear_history()
    await update.message.reply_text("✅ 对话历史已清除")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /status 命令"""
    if streaming_processor is None:
        await update.message.reply_text("❌ 系统未初始化")
        return
    
    stats = streaming_processor.engine.get_statistics()
    
    status = f"""
📊 *系统状态*

*流式引擎:*
• 总Token数: {stats['total_tokens']}
• 平均Token时间: {stats['avg_token_time_ms']:.2f}ms
• 实际刷新率: {stats['refresh_rate_hz']:.1f}Hz

*配置:*
• 目标刷新率: {REFRESH_RATE_HZ}Hz
• 刷新周期: {REFRESH_CYCLE_MS}ms
"""
    await update.message.reply_text(status, parse_mode='Markdown')


async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /train 命令"""
    if brain is None:
        await update.message.reply_text("❌ 架构未初始化")
        return
    
    await update.message.reply_text("⏳ 开始训练...")
    
    training_data = [
        {"input": "你好", "output": "你好！我是类人脑AI助手。"},
        {"input": "你是谁", "output": "我是基于类人脑双系统架构的AI。"},
        {"input": "什么是STDP", "output": "STDP是脉冲时序依赖可塑性学习规则。"},
    ]
    
    result = brain.train(training_data, epochs=2)
    
    if result.get("success"):
        await update.message.reply_text(
            f"✅ 训练完成\n"
            f"轮次: {len(result.get('training_log', []))}\n"
            f"耗时: {result.get('total_time', 0):.1f}秒"
        )


# ==================== 消息处理 ====================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理普通消息 - 流式输出"""
    if streaming_processor is None:
        await update.message.reply_text("❌ 系统未初始化")
        return
    
    user_input = update.message.text
    chat_id = update.effective_chat.id
    
    # 发送"正在输入"状态
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    try:
        # 使用流式处理器
        final_text, stats = await streaming_processor.process_message(
            context.bot,
            chat_id,
            user_input
        )
        
        # 发送统计信息
        engine_stats = stats.get('engine', {})
        streamer_stats = stats.get('streamer', {})
        
        info = (
            f"📊 Token: {engine_stats.get('total_tokens', 0)} | "
            f"刷新率: {engine_stats.get('refresh_rate_hz', 0):.1f}Hz | "
            f"更新次数: {streamer_stats.get('total_updates', 0)}"
        )
        await update.message.reply_text(info)
        
    except Exception as e:
        logger.error(f"处理消息错误: {e}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text(f"❌ 错误: {str(e)}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    if not TELEGRAM_AVAILABLE:
        print("请先安装: pip install python-telegram-bot")
        return
    
    # 初始化
    if not initialize():
        print("初始化失败，Bot将以有限功能运行")
        return
    
    # 配置请求超时
    request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=60.0,
        write_timeout=30.0,
        pool_timeout=30.0
    )
    
    # 创建应用
    application = Application.builder().token(TOKEN).request(request).build()
    
    # 添加处理器
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("train", train_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # 启动
    logger.info(f"启动Telegram Bot...")
    logger.info(f"Bot Token: {TOKEN[:20]}...")
    logger.info(f"请在Telegram中搜索Bot开始对话")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
