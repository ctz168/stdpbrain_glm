#!/bin/bash

# ============================================================
# 类人脑双系统AI架构 - 启动脚本
# Brain Architecture - Start Script
# ============================================================

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "================================================================"
echo "      类人脑双系统AI架构 - 启动服务"
echo "================================================================"
echo -e "${NC}"

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}[OK]${NC} 虚拟环境已激活"
else
    echo -e "${YELLOW}[WARN]${NC} 虚拟环境不存在，使用系统Python"
fi

# 加载环境变量
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}[OK]${NC} 环境变量已加载"
fi

# 创建日志目录
mkdir -p logs

# 获取端口
WEB_PORT=${WEB_PORT:-3000}
BOT_PORT=${BOT_PORT:-8080}

echo ""
echo -e "${CYAN}[1/2]${NC} 启动Web服务 (端口: ${WEB_PORT})..."
python -m uvicorn src.api.main:app --host 0.0.0.0 --port $WEB_PORT > logs/web.log 2>&1 &
WEB_PID=$!
echo -e "${GREEN}[OK]${NC} Web服务已启动 (PID: $WEB_PID)"

sleep 2

echo ""
echo -e "${CYAN}[2/2]${NC} 启动Telegram Bot..."
python src/bot/telegram_bot.py > logs/bot.log 2>&1 &
BOT_PID=$!
echo -e "${GREEN}[OK]${NC} Telegram Bot已启动 (PID: $BOT_PID)"

echo ""
echo -e "${GREEN}"
echo "================================================================"
echo "                    服务已启动!"
echo "================================================================"
echo -e "${NC}"
echo ""
echo -e "  ${CYAN}Web界面:${NC}    http://localhost:${WEB_PORT}"
echo -e "  ${CYAN}API文档:${NC}    http://localhost:${WEB_PORT}/docs"
echo -e "  ${CYAN}Telegram Bot:${NC} 已启动，请在Telegram中与Bot对话"
echo ""
echo -e "  ${YELLOW}日志目录:${NC}    ./logs/"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止所有服务${NC}"
echo ""

# 保存PID
echo $WEB_PID > logs/web.pid
echo $BOT_PID > logs/bot.pid

# 等待中断信号
trap "echo ''; echo '停止所有服务...'; kill $WEB_PID $BOT_PID 2>/dev/null; rm -f logs/*.pid; echo '服务已停止'; exit 0" SIGINT SIGTERM

# 等待
wait
