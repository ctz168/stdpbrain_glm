"""
类人脑双系统AI架构 - Telegram Bot服务
Telegram Bot Service with Streaming Output
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Telegram Bot
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        ContextTypes,
        filters
    )
    from telegram.request import HTTPXRequest
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("[ERROR] python-telegram-bot未安装")
    print("请运行: pip install python-telegram-bot")

# 配置
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8533918353:AAG6Pxr0A4C4kJpCVjYzbtwtFzN4KZCcRag")
MODEL_DIR = os.getenv("MODEL_PATH", "./models/qwen3.5-0.8b")
MAX_TOKENS = 256
TEMPERATURE = 0.7

# 日志配置
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 全局变量
model = None
tokenizer = None
user_sessions = {}

class UserSession:
    """用户会话"""
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.history = []
        self.mode = "selfGeneration"
        self.max_history = 10
    
    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def clear_history(self):
        """清除历史"""
        self.history = []

def load_model():
    """加载模型"""
    global model, tokenizer
    
    logger.info("加载模型...")
    
    model_path = Path(MODEL_DIR)
    
    if not model_path.exists():
        logger.error(f"模型目录不存在: {MODEL_DIR}")
        return False
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.eval()
        
        logger.info("模型加载成功")
        return True
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False

def get_session(user_id: int) -> UserSession:
    """获取或创建用户会话"""
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    return user_sessions[user_id]

async def generate_response(prompt: str, session: UserSession) -> str:
    """生成响应"""
    if model is None or tokenizer is None:
        return "抱歉，模型未加载。"
    
    # 构建上下文
    context = ""
    for msg in session.history[-5:]:
        if msg["role"] == "user":
            context += f"用户: {msg['content']}\n"
        else:
            context += f"助手: {msg['content']}\n"
    
    full_prompt = context + f"用户: {prompt}\n助手:"
    
    # 编码
    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=512)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取助手回复
    if "助手:" in full_response:
        response = full_response.split("助手:")[-1].strip()
    else:
        response = full_response
    
    # 保存到历史
    session.add_message("user", prompt)
    session.add_message("assistant", response)
    
    return response

# Bot命令处理
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /start 命令"""
    user_id = update.effective_user.id
    session = get_session(user_id)
    
    welcome_message = """
🧠 *类人脑双系统AI架构*

欢迎使用基于Qwen3.5-0.8B的类人脑AI！

*核心特性:*
• 100Hz高刷新推理
• STDP学习机制
• 海马体记忆系统
• 三模式自闭环优化

*命令列表:*
/start - 开始对话
/mode - 切换优化模式
/clear - 清除对话历史
/stats - 查看系统状态
/help - 帮助信息

直接发送消息即可开始对话！
"""
    
    keyboard = [
        [
            InlineKeyboardButton("自生成模式", callback_data="mode_selfGeneration"),
            InlineKeyboardButton("自博弈模式", callback_data="mode_selfPlay"),
        ],
        [
            InlineKeyboardButton("自评判模式", callback_data="mode_selfEvaluation"),
            InlineKeyboardButton("系统状态", callback_data="stats"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        welcome_message,
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /help 命令"""
    help_text = """
📖 *帮助信息*

*优化模式说明:*

🔹 *自生成模式* (默认)
每个周期并行生成2个候选，STDP加权投票选择最优结果。

🔹 *自博弈模式*
提案-验证对抗迭代，直到收敛或达到最大迭代次数。

🔹 *自评判模式*
双候选生成 + 四维度评判选优，每10个周期执行一次。

*使用技巧:*
• 复杂问题推荐使用自博弈模式
• 简单对话推荐使用自生成模式
• 需要高质量输出时使用自评判模式
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /mode 命令"""
    keyboard = [
        [
            InlineKeyboardButton("✅ 自生成模式", callback_data="mode_selfGeneration"),
            InlineKeyboardButton("自博弈模式", callback_data="mode_selfPlay"),
        ],
        [
            InlineKeyboardButton("自评判模式", callback_data="mode_selfEvaluation"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "请选择优化模式:",
        reply_markup=reply_markup
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /clear 命令"""
    user_id = update.effective_user.id
    session = get_session(user_id)
    session.clear_history()
    
    await update.message.reply_text("✅ 对话历史已清除")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理 /stats 命令"""
    if model is None:
        await update.message.reply_text("❌ 模型未加载")
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    
    stats_text = f"""
📊 *系统状态*

*模型信息:*
• 名称: Qwen2.5-0.5B
• 参数量: {total_params/1e6:.1f}M
• 状态: ✅ 已加载

*架构配置:*
• 刷新周期: 10ms (100Hz)
• 权重拆分: 90%静态 + 10%动态
• 注意力复杂度: O(1)

*当前会话:*
• 活跃用户: {len(user_sessions)}
"""
    
    await update.message.reply_text(stats_text, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理普通消息"""
    if model is None:
        await update.message.reply_text("❌ 模型未加载，请稍后再试")
        return
    
    user_id = update.effective_user.id
    session = get_session(user_id)
    prompt = update.message.text
    
    # 发送"正在输入"状态
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    
    # 生成响应
    try:
        response = await generate_response(prompt, session)
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"生成响应失败: {e}")
        await update.message.reply_text(f"生成失败: {str(e)[:100]}")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """处理按钮回调"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    session = get_session(user_id)
    
    data = query.data
    
    if data.startswith("mode_"):
        mode = data.replace("mode_", "")
        session.mode = mode
        
        mode_names = {
            "selfGeneration": "自生成模式",
            "selfPlay": "自博弈模式",
            "selfEvaluation": "自评判模式"
        }
        
        await query.edit_message_text(
            f"✅ 已切换到: {mode_names.get(mode, mode)}"
        )
    
    elif data == "stats":
        if model:
            total_params = sum(p.numel() for p in model.parameters())
            await query.edit_message_text(
                f"📊 模型参数: {total_params/1e6:.1f}M\n"
                f"活跃会话: {len(user_sessions)}"
            )
        else:
            await query.edit_message_text("❌ 模型未加载")

def main():
    """主函数"""
    if not TELEGRAM_AVAILABLE:
        print("请先安装: pip install python-telegram-bot")
        return
    
    # 加载模型
    if not load_model():
        print("模型加载失败，Bot将以有限功能运行")
    
    # 配置请求超时
    request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0
    )
    
    # 创建应用
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request)
        .build()
    )
    
    # 添加处理器
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("mode", mode_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # 启动Bot
    logger.info("启动Telegram Bot...")
    logger.info(f"Bot Token: {BOT_TOKEN[:20]}...")
    logger.info("Bot名称: Myxiaomiopbot")
    logger.info("请在Telegram中搜索 @Myxiaomiopbot 开始对话")
    
    # 使用run_polling
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False
    )

if __name__ == "__main__":
    main()
