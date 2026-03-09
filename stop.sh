#!/bin/bash

# ============================================================
# 类人脑双系统AI架构 - 停止脚本
# Brain Architecture - Stop Script
# ============================================================

echo "停止所有服务..."

# 读取PID并停止
if [ -f "logs/web.pid" ]; then
    WEB_PID=$(cat logs/web.pid)
    kill $WEB_PID 2>/dev/null
    rm -f logs/web.pid
    echo "  ✓ Web服务已停止"
fi

if [ -f "logs/bot.pid" ]; then
    BOT_PID=$(cat logs/bot.pid)
    kill $BOT_PID 2>/dev/null
    rm -f logs/bot.pid
    echo "  ✓ Telegram Bot已停止"
fi

# 确保所有相关进程都停止
pkill -f "uvicorn src.api.main" 2>/dev/null
pkill -f "telegram_bot.py" 2>/dev/null

echo "所有服务已停止"
