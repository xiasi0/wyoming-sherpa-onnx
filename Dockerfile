# 多阶段构建 - 构建阶段
FROM python:3.12-slim AS builder

WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 运行阶段
FROM python:3.12-slim

LABEL maintainer="Wyoming Sherpa ONNX"
LABEL description="Wyoming ASR server with sherpa-onnx"

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY app/ ./app/
COPY run_server.py .

# 创建模型目录和数据目录
RUN mkdir -p /data/models

# 清理构建缓存
RUN rm -rf /var/cache/apt/* /var/log/dpkg.log /var/log/apt/* /tmp/*

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /data/models && \
    chown -R appuser:appuser /data && \
    rm -rf /root/.cache /tmp/*

USER appuser

# 暴露端口
EXPOSE 10300

# 环境变量默认值
ENV HOST=0.0.0.0
ENV PORT=10300
ENV SERVICE_NAME=wyoming-sherpa-onnx
ENV ZEROCONF=false
ENV MODEL_NAME=sherpa-onnx-funasr-nano-int8-2025-12-30
ENV MODEL_DIR=/data/models/sherpa-onnx-funasr-nano-int8-2025-12-30
ENV SAMPLE_RATE=16000
ENV NUM_THREADS=2
ENV AUTO_DOWNLOAD=true
ENV DEBUG=false

# 启动命令
ENTRYPOINT ["python", "run_server.py"]
