#!/bin/bash

# ============================================================
# 类人脑双系统AI架构 - 一键安装脚本
# Brain Architecture - One-Click Installation Script
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logo
echo -e "${PURPLE}"
echo "================================================================"
echo "      类人脑双系统AI架构 - 一键安装"
echo "      Brain Architecture - One-Click Installation"
echo "================================================================"
echo -e "${NC}"

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get &> /dev/null; then
            PKG_MANAGER="apt"
        elif command -v yum &> /dev/null; then
            PKG_MANAGER="yum"
        elif command -v dnf &> /dev/null; then
            PKG_MANAGER="dnf"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        PKG_MANAGER="choco"
    else
        OS="unknown"
    fi
    echo -e "${CYAN}[INFO]${NC} 检测到操作系统: ${YELLOW}$OS${NC}"
}

# 检查Python版本
check_python() {
    echo -e "${CYAN}[1/8]${NC} 检查Python环境..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}[ERROR]${NC} 未找到Python，请先安装Python 3.10+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}[OK]${NC} Python版本: ${YELLOW}$PYTHON_VERSION${NC}"
    
    # 检查Python版本是否>=3.10
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $MAJOR -lt 3 ]] || [[ $MAJOR -eq 3 && $MINOR -lt 10 ]]; then
        echo -e "${RED}[ERROR]${NC} Python版本过低，需要3.10+"
        exit 1
    fi
}

# 创建虚拟环境
create_venv() {
    echo -e "${CYAN}[2/8]${NC} 创建虚拟环境..."
    
    if [ -d "venv" ]; then
        echo -e "${YELLOW}[WARN]${NC} 虚拟环境已存在，跳过创建"
    else
        $PYTHON_CMD -m venv venv
        echo -e "${GREEN}[OK]${NC} 虚拟环境创建成功"
    fi
    
    # 激活虚拟环境
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    echo -e "${GREEN}[OK]${NC} 虚拟环境已激活"
}

# 安装系统依赖
install_system_deps() {
    echo -e "${CYAN}[3/8]${NC} 安装系统依赖..."
    
    if [[ "$OS" == "linux" ]]; then
        if [[ "$PKG_MANAGER" == "apt" ]]; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq build-essential python3-dev
        elif [[ "$PKG_MANAGER" == "yum" ]]; then
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y python3-devel
        fi
    elif [[ "$OS" == "macos" ]]; then
        if ! command -v brew &> /dev/null; then
            echo -e "${YELLOW}[WARN]${NC} Homebrew未安装，跳过系统依赖"
        else
            brew install python@3.10 || true
        fi
    fi
    
    echo -e "${GREEN}[OK]${NC} 系统依赖安装完成"
}

# 安装Python依赖
install_python_deps() {
    echo -e "${CYAN}[4/8]${NC} 安装Python依赖..."
    
    # 升级pip
    pip install --upgrade pip -q
    
    # 安装PyTorch (CPU版本，更小)
    echo -e "${YELLOW}[INFO]${NC} 安装PyTorch (CPU版本)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q
    
    # 安装其他依赖
    echo -e "${YELLOW}[INFO]${NC} 安装其他依赖..."
    pip install -r requirements.txt -q
    
    echo -e "${GREEN}[OK]${NC} Python依赖安装完成"
}

# 创建必要目录
create_dirs() {
    echo -e "${CYAN}[5/8]${NC} 创建目录结构..."
    
    mkdir -p models/qwen3.5-0.8b
    mkdir -p models/trained
    mkdir -p models/cache
    mkdir -p logs
    mkdir -p data
    
    echo -e "${GREEN}[OK]${NC} 目录创建完成"
}

# 配置环境变量
setup_env() {
    echo -e "${CYAN}[6/8]${NC} 配置环境变量..."
    
    if [ -f ".env" ]; then
        echo -e "${YELLOW}[WARN]${NC} .env文件已存在，跳过"
    else
        cp .env.example .env
        echo -e "${GREEN}[OK]${NC} .env文件已创建，请根据需要修改配置"
    fi
}

# 下载模型
download_model() {
    echo -e "${CYAN}[7/8]${NC} 下载模型..."
    
    if [ -f "models/qwen3.5-0.8b/model.safetensors" ]; then
        echo -e "${YELLOW}[WARN]${NC} 模型已存在，跳过下载"
    else
        echo -e "${YELLOW}[INFO]${NC} 开始下载Qwen2.5-0.5B模型..."
        echo -e "${YELLOW}[INFO]${NC} 这可能需要几分钟，请耐心等待..."
        
        python scripts/download_model.py
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[OK]${NC} 模型下载完成"
        else
            echo -e "${RED}[ERROR]${NC} 模型下载失败，请检查网络连接"
            echo -e "${YELLOW}[INFO]${NC} 您可以稍后手动运行: python scripts/download_model.py"
        fi
    fi
}

# 创建启动脚本
create_start_script() {
    echo -e "${CYAN}[8/8]${NC} 创建启动脚本..."
    
    # 创建启动脚本
    cat > start.sh << 'EOF'
#!/bin/bash
source venv/bin/activate

echo "=========================================="
echo "  类人脑双系统AI架构 - 启动服务"
echo "=========================================="

# 启动Web服务
echo "[1/2] 启动Web服务 (端口: 3000)..."
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 3000 &
WEB_PID=$!

# 等待Web服务启动
sleep 3

# 启动Telegram Bot
echo "[2/2] 启动Telegram Bot..."
python src/bot/telegram_bot.py &
BOT_PID=$!

echo ""
echo "=========================================="
echo "  服务已启动!"
echo "  Web界面: http://localhost:3000"
echo "  API文档: http://localhost:3000/docs"
echo "=========================================="
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待中断信号
trap "kill $WEB_PID $BOT_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
EOF
    
    chmod +x start.sh
    
    # 创建停止脚本
    cat > stop.sh << 'EOF'
#!/bin/bash
echo "停止所有服务..."
pkill -f "uvicorn src.api.main"
pkill -f "telegram_bot.py"
echo "服务已停止"
EOF
    
    chmod +x stop.sh
    
    # 创建训练脚本
    cat > train.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python scripts/train.py "$@"
EOF
    
    chmod +x train.sh
    
    echo -e "${GREEN}[OK]${NC} 启动脚本创建完成"
}

# 安装完成
install_complete() {
    echo ""
    echo -e "${GREEN}"
    echo "================================================================"
    echo "                    安装完成！"
    echo "================================================================"
    echo -e "${NC}"
    echo ""
    echo -e "下一步操作:"
    echo ""
    echo -e "  ${CYAN}1.${NC} 编辑配置文件 (可选):"
    echo -e "      ${YELLOW}nano .env${NC}"
    echo ""
    echo -e "  ${CYAN}2.${NC} 启动服务:"
    echo -e "      ${YELLOW}./start.sh${NC}"
    echo ""
    echo -e "  ${CYAN}3.${NC} 访问Web界面:"
    echo -e "      ${YELLOW}http://localhost:3000${NC}"
    echo ""
    echo -e "  ${CYAN}4.${NC} 使用Telegram Bot:"
    echo -e "      在Telegram中搜索您的Bot开始对话"
    echo ""
    echo -e "  ${CYAN}5.${NC} 执行训练:"
    echo -e "      ${YELLOW}./train.sh --scenario dialogue --epochs 3${NC}"
    echo ""
    echo -e "${PURPLE}感谢使用类人脑双系统AI架构！${NC}"
    echo ""
}

# 主函数
main() {
    detect_os
    check_python
    create_venv
    install_system_deps
    install_python_deps
    create_dirs
    setup_env
    download_model
    create_start_script
    install_complete
}

# 运行主函数
main
