# 类人脑双系统AI架构 - Dockerfile
# Brain Architecture Docker Image

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装PyTorch CPU版本
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 复制项目文件
COPY . .

# 创建必要目录
RUN mkdir -p models logs data

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/qwen3.5-0.8b

# 暴露端口
EXPOSE 3000 8080

# 启动命令
CMD ["./start.sh"]
