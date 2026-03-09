#!/bin/bash

# ============================================================
# 类人脑双系统AI架构 - 训练脚本
# Brain Architecture - Training Script
# ============================================================

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 加载环境变量
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 默认参数
SCENARIO=${1:-"dialogue"}
EPOCHS=${2:-3}

echo "=========================================="
echo "  类人脑双系统AI架构 - 训练"
echo "=========================================="
echo ""
echo "场景: $SCENARIO"
echo "轮次: $EPOCHS"
echo ""

python scripts/lightweight_training.py --scenario $SCENARIO --epochs $EPOCHS
