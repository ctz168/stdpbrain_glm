#!/usr/bin/env python3
"""
类人脑双系统AI架构 - Telegram Bot服务
使用生产级核心架构
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.brain_architecture import BrainArchitecture, BrainConfig

# Telegram Bot
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        CallbackQueryHandler, ContextTypes, filters
    )
    from telegram.request import HTTPXRequest
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("[ERROR] python-telegram-bot未安装")

# 配置
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8533918353:AAG6Pxr0A4C4kJpCVjYzbtwtFzN4KZCcRag")
MODEL_DIR = os.getenv("MODEL_PATH", "./models/qwen3.5-0.8b")

# 日志
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局架构实例
brain = None

def initialize_brain():
    """初始化架构"""
    global brain
    logger.info("初始化类人脑双系统AI架构...")
    
    config = BrainConfig(model_path=MODEL_DIR)
    brain = BrainArchitecture(config)
    result = brain.initialize()
    
    if result["success"]:
        logger.info(f"架构初始化成功: {result['model_info']['name']}")
        return True
    else:
        logger.error(f"架构初始化失败: {result['message']}")
        return False


# ==================== 命令处理 ====================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /start 命令"""
    welcome = """
🧠 *类人脑双系统AI架构*

欢迎使用基于Qwen2.5-0.5B的类人脑AI！

*核心特性:*
• 100Hz高刷新推理 (10ms周期)
• STDP学习机制 (无反向传播)
• 海马体记忆系统 (EC-DG-CA3-CA1)
• 权重双轨拆分 (90%静态+10%动态)

*命令列表:*
/start - 开始对话
/clear - 清除对话历史
/status - 查看系统状态
/train - 执行训练
/help - 帮助信息

直接发送消息即可开始对话！
"""
    keyboard = [
        [InlineKeyboardButton("清除历史", callback_data="clear"),
         InlineKeyboardButton("系统状态", callback_data="status")],
        [InlineKeyboardButton("执行训练", callback_data="train")]
    ]
    await update.message.reply_text(welcome, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /help 命令"""
    help_text = """
📖 *帮助信息*

*架构说明:*
本系统实现了完整的类人脑双系统AI架构：

1. *100Hz推理引擎* - 每10ms完成一次推理周期
2. *STDP学习* - 基于时序的突触可塑性学习
3. *海马体记忆* - EC→DG→CA3→CA1记忆编码
4. *权重双轨* - 90%静态冻结 + 10%动态更新

*使用技巧:*
• 支持多轮对话，自动保持上下文
• 每次对话都会更新海马体记忆
• STDP权重实时更新
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /clear 命令"""
    if brain:
        brain.clear_history()
    await update.message.reply_text("✅ 对话历史已清除")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /status 命令"""
    if not brain:
        await update.message.reply_text("❌ 架构未初始化")
        return
    
    status = brain.get_status()
    metrics = status.get("metrics", {})
    stdp_stats = status.get("stdp_stats", {})
    hippocampus_stats = status.get("hippocampus_stats", {})
    
    status_text = f"""
📊 *系统状态*

*架构状态:*
• 初始化: {'✅' if status.get('initialized') else '❌'}
• 运行中: {'✅' if status.get('is_running') else '❌'}
• 周期数: {status.get('cycle_count', 0)}

*性能指标:*
• 平均周期时间: {metrics.get('avg_cycle_time', 0):.2f}ms
• 最大周期时间: {metrics.get('max_cycle_time', 0):.2f}ms

*STDP统计:*
• 总更新: {stdp_stats.get('total_updates', 0)}
• LTP: {stdp_stats.get('ltp_count', 0)} | LTD: {stdp_stats.get('ltd_count', 0)}

*海马体记忆:*
• 总记忆: {hippocampus_stats.get('total_memories', 0)}
• 平均强度: {hippocampus_stats.get('avg_strength', 0):.2f}
"""
    await update.message.reply_text(status_text, parse_mode='Markdown')


async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /train 命令"""
    if not brain:
        await update.message.reply_text("❌ 架构未初始化")
        return
    
    await update.message.reply_text("⏳ 开始训练...")
    
    # 训练数据
    training_data = [
        {"input": "你好", "output": "你好！我是类人脑AI助手，很高兴为您服务。"},
        {"input": "你是谁", "output": "我是基于类人脑双系统架构的AI，具有STDP学习能力和海马体记忆系统。"},
        {"input": "什么是STDP？", "output": "STDP是脉冲时序依赖可塑性，一种基于时序的突触可塑性学习规则。"},
        {"input": "1+1等于几？", "output": "1+1=2"},
    ]
    
    result = brain.train(training_data, epochs=2)
    
    if result.get("success"):
        log = result.get("training_log", [])
        final_loss = log[-1].get("avg_loss", 0) if log else 0
        await update.message.reply_text(
            f"✅ 训练完成\n"
            f"轮次: {len(log)}\n"
            f"最终损失: {final_loss:.4f}\n"
            f"耗时: {result.get('total_time', 0):.1f}秒"
        )
    else:
        await update.message.reply_text(f"❌ 训练失败: {result.get('error', '未知错误')}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理普通消息"""
    if not brain:
        await update.message.reply_text("❌ 架构未初始化，请稍后再试")
        return
    
    user_input = update.message.text
    
    # 发送"正在输入"状态
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # 执行推理
    result = brain.infer(user_input)
    
    if "error" in result:
        await update.message.reply_text(f"错误: {result['error']}")
        return
    
    output = result.get("output", "")
    metadata = result.get("metadata", {})
    
    # 发送响应
    await update.message.reply_text(output)
    
    # 发送元数据
    if metadata:
        info = f"\n📊 周期: {metadata.get('cycle_id', 0)} | 耗时: {metadata.get('cycle_time_ms', 0):.1f}ms"
        await update.message.reply_text(info)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理按钮回调"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "clear":
        if brain:
            brain.clear_history()
        await query.edit_message_text("✅ 对话历史已清除")
    
    elif data == "status":
        if brain:
            status = brain.get_status()
            await query.edit_message_text(f"📊 周期数: {status.get('cycle_count', 0)}")
        else:
            await query.edit_message_text("❌ 架构未初始化")
    
    elif data == "train":
        await query.edit_message_text("⏳ 请使用 /train 命令开始训练")


def main():
    """主函数"""
    if not TELEGRAM_AVAILABLE:
        print("请先安装: pip install python-telegram-bot")
        return
    
    # 初始化架构
    if not initialize_brain():
        print("架构初始化失败，Bot将以有限功能运行")
    
    # 配置请求超时
    request = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0, write_timeout=30.0, pool_timeout=30.0)
    
    # 创建应用
    application = Application.builder().token(BOT_TOKEN).request(request).build()
    
    # 添加处理器
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("train", train_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # 启动Bot
    logger.info("启动Telegram Bot...")
    logger.info(f"Bot Token: {BOT_TOKEN[:20]}...")
    logger.info("Bot名称: Myxiaomiopbot")
    logger.info("请在Telegram中搜索 @Myxiaomiopbot 开始对话")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
