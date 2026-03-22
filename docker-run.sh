#!/bin/bash
#
# Wyoming Sherpa FunASR Docker 启动脚本
# 使用环境变量配置容器参数
#

set -e

# 默认配置
WYOMING_HOST="${WYOMING_HOST:-0.0.0.0}"
WYOMING_PORT="${WYOMING_PORT:-10300}"
WYOMING_SERVICE_NAME="${WYOMING_SERVICE_NAME:-sherpa-funasr}"
WYOMING_ZEROCONF="${WYOMING_ZEROCONF:-true}"
WYOMING_MODEL_NAME="${WYOMING_MODEL_NAME:-sherpa-onnx-funasr-nano-int8-2025-12-30}"
WYOMING_MODEL_DIR="${WYOMING_MODEL_DIR:-/data/models/sherpa-onnx-funasr-nano-int8-2025-12-30}"
WYOMING_SAMPLE_RATE="${WYOMING_SAMPLE_RATE:-16000}"
WYOMING_NUM_THREADS="${WYOMING_NUM_THREADS:-2}"
WYOMING_AUTO_DOWNLOAD="${WYOMING_AUTO_DOWNLOAD:-true}"
WYOMING_DEBUG="${WYOMING_DEBUG:-false}"

# 容器配置
CONTAINER_NAME="${CONTAINER_NAME:-wyoming-sherpa-funasr}"
IMAGE_NAME="${IMAGE_NAME:-wyoming-sherpa-funasr:latest}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/data/models}"
CONTAINER_MODEL_DIR="/data/models"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_IMAGE=true
            shift
            ;;
        --no-zeroconf)
            WYOMING_ZEROCONF=false
            shift
            ;;
        --debug)
            WYOMING_DEBUG=true
            shift
            ;;
        --model-dir)
            HOST_MODEL_DIR="$2"
            WYOMING_MODEL_DIR="/data/models/$(basename "$2")"
            shift 2
            ;;
        --port)
            WYOMING_PORT="$2"
            shift 2
            ;;
        --help)
            echo "用法：$0 [选项]"
            echo ""
            echo "选项:"
            echo "  --build           构建镜像"
            echo "  --no-zeroconf     禁用 Zeroconf/mDNS"
            echo "  --debug           启用调试日志"
            echo "  --model-dir DIR   宿主机模型目录 (默认：/data/models)"
            echo "  --port PORT       服务端口 (默认：10300)"
            echo "  --help            显示帮助信息"
            echo ""
            echo "环境变量:"
            echo "  WYOMING_HOST          监听地址 (默认：0.0.0.0)"
            echo "  WYOMING_PORT          监听端口 (默认：10300)"
            echo "  WYOMING_SERVICE_NAME  服务名称 (默认：sherpa-funasr)"
            echo "  WYOMING_ZEROCONF      启用 Zeroconf (默认：true)"
            echo "  WYOMING_MODEL_NAME    模型名称 (默认：sherpa-onnx-funasr-nano-int8-2025-12-30)"
            echo "  WYOMING_MODEL_DIR     容器内模型目录 (默认：/data/models/sherpa-onnx-funasr-nano-int8-2025-12-30)"
            echo "  WYOMING_SAMPLE_RATE   采样率 (默认：16000)"
            echo "  WYOMING_NUM_THREADS   线程数 (默认：2)"
            echo "  WYOMING_AUTO_DOWNLOAD 自动下载模型 (默认：true)"
            echo "  WYOMING_DEBUG         调试模式 (默认：false)"
            echo "  HOST_MODEL_DIR        宿主机模型目录 (默认：/data/models)"
            echo "  CONTAINER_NAME        容器名称 (默认：wyoming-sherpa-funasr)"
            echo "  IMAGE_NAME            镜像名称 (默认：wyoming-sherpa-funasr:latest)"
            exit 0
            ;;
        *)
            echo "未知选项：$1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 构建镜像
if [[ "${BUILD_IMAGE}" == "true" ]]; then
    echo "正在构建 Docker 镜像..."
    docker build -t "$IMAGE_NAME" .
fi

# 检查镜像是否存在
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "镜像不存在，正在构建..."
    docker build -t "$IMAGE_NAME" .
fi

# 停止并删除旧容器
echo "清理旧容器..."
docker rm -f "$CONTAINER_NAME" &>/dev/null || true

# 构建 Zeroconf 参数
ZEROCONF_ARG=""
if [[ "${WYOMING_ZEROCONF}" == "false" ]]; then
    ZEROCONF_ARG="--no-zeroconf"
fi

# 构建调试参数
DEBUG_ARG=""
if [[ "${WYOMING_DEBUG}" == "true" ]]; then
    DEBUG_ARG="--debug"
fi

# 创建宿主机模型目录
echo "创建宿主机模型目录：$HOST_MODEL_DIR"
mkdir -p "$HOST_MODEL_DIR"

# 启动容器
echo "启动 Wyoming Sherpa FunASR 容器..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p "$WYOMING_PORT:$WYOMING_PORT" \
    -v "$HOST_MODEL_DIR:$CONTAINER_MODEL_DIR:rw" \
    -e WYOMING_HOST="$WYOMING_HOST" \
    -e WYOMING_PORT="$WYOMING_PORT" \
    -e WYOMING_SERVICE_NAME="$WYOMING_SERVICE_NAME" \
    -e WYOMING_MODEL_NAME="$WYOMING_MODEL_NAME" \
    -e WYOMING_MODEL_DIR="$WYOMING_MODEL_DIR" \
    -e WYOMING_SAMPLE_RATE="$WYOMING_SAMPLE_RATE" \
    -e WYOMING_NUM_THREADS="$WYOMING_NUM_THREADS" \
    -e WYOMING_AUTO_DOWNLOAD="$WYOMING_AUTO_DOWNLOAD" \
    -e WYOMING_DEBUG="$WYOMING_DEBUG" \
    "$IMAGE_NAME" \
    --host "$WYOMING_HOST" \
    --port "$WYOMING_PORT" \
    --service-name "$WYOMING_SERVICE_NAME" \
    --model-dir "$WYOMING_MODEL_DIR" \
    --model-name "$WYOMING_MODEL_NAME" \
    --sample-rate "$WYOMING_SAMPLE_RATE" \
    --num-threads "$WYOMING_NUM_THREADS" \
    $ZEROCONF_ARG \
    $DEBUG_ARG

echo ""
echo "容器已启动："
echo "  容器名称：$CONTAINER_NAME"
echo "  镜像：$IMAGE_NAME"
echo "  监听地址：$WYOMING_HOST:$WYOMING_PORT"
echo "  模型目录：$HOST_MODEL_DIR -> $CONTAINER_MODEL_DIR"
echo ""
echo "查看日志：docker logs -f $CONTAINER_NAME"
echo "停止服务：docker stop $CONTAINER_NAME"
echo "重启服务：docker restart $CONTAINER_NAME"
