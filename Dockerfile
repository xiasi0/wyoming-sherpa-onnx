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

LABEL maintainer="Wyoming Sherpa FunASR"
LABEL description="Wyoming ASR server with sherpa-onnx FunASR Nano"

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

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app /data
USER appuser

# 暴露端口
EXPOSE 10300

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('127.0.0.1', 10300)); s.close()" || exit 1

# 环境变量默认值
ENV WYOMING_HOST=0.0.0.0
ENV WYOMING_PORT=10300
ENV WYOMING_SERVICE_NAME=sherpa-funasr
ENV WYOMING_ZEROCONF=true
ENV WYOMING_MODEL_NAME=sherpa-onnx-funasr-nano-int8-2025-12-30
ENV WYOMING_MODEL_DIR=/data/models/sherpa-onnx-funasr-nano-int8-2025-12-30
ENV WYOMING_SAMPLE_RATE=16000
ENV WYOMING_NUM_THREADS=2
ENV WYOMING_AUTO_DOWNLOAD=true
ENV WYOMING_DEBUG=false

# 启动命令
ENTRYPOINT ["python", "run_server.py"]
CMD [
    "--host", "${WYOMING_HOST}",
    "--port", "${WYOMING_PORT}",
    "--service-name", "${WYOMING_SERVICE_NAME}",
    "--model-dir", "${WYOMING_MODEL_DIR}",
    "--model-name", "${WYOMING_MODEL_NAME}",
    "--sample-rate", "${WYOMING_SAMPLE_RATE}",
    "--num-threads", "${WYOMING_NUM_THREADS}"
]
