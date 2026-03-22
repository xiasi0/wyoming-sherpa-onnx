# Wyoming + sherpa-onnx (FunASR Nano) ASR Server

这个项目实现了一个 **Wyoming 协议服务端**，底层使用 `sherpa-onnx-funasr-nano-int8-2025-12-30` 做离线语音识别，并支持通过 mDNS 广播 `_wyoming._tcp` 供 Home Assistant 自动发现。

## 1. 功能特性

- 支持 Wyoming 基础 ASR 流程：`describe/info`、`transcribe`、`audio-start/chunk/stop`、`transcript`
- 模型固定为 `sherpa-onnx-funasr-nano-int8-2025-12-30`（可改 `--model-name` 仅影响暴露名）
- 支持 Zeroconf/mDNS 广播，便于 HA 自动发现
- 支持 16-bit PCM 输入，多声道自动混合单声道，非 16k 会线性重采样到 16k
- 支持 Docker 部署，模型数据持久化

## 2. 目录结构

```text
wyoming-sherpa-funasr/
  app/
    asr_engine.py
    config.py
    discovery.py
    protocol.py
    server.py
    downloader.py
  data/                    # 模型数据目录（Docker 卷映射）
  Dockerfile
  docker-compose.yml
  docker-run.sh
  run_server.py
  requirements.txt
```

## 3. Docker 部署（推荐）

### 3.1 使用 docker-compose（推荐）

```bash
cd /wyoming-sherpa-funasr

# 构建并启动
docker-compose up -d --build

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 3.2 使用启动脚本

```bash
cd /wyoming-sherpa-funasr

# 基本启动（自动构建镜像）
./docker-run.sh --build

# 禁用 Zeroconf
./docker-run.sh --no-zeroconf

# 自定义模型目录和端口
./docker-run.sh --model-dir /data/models --port 10300

# 启用调试模式
./docker-run.sh --debug

# 查看帮助
./docker-run.sh --help
```

### 3.3 使用 docker run 命令

```bash
# 创建模型目录
mkdir -p /data/models

# 启动容器
docker run -d \
  --name wyoming-sherpa-funasr \
  --restart unless-stopped \
  -p 10300:10300 \
  -v /data/models:/data/models:rw \
  -e WYOMING_PORT=10300 \
  -e WYOMING_ZEROCONF=true \
  -e WYOMING_MODEL_DIR=/data/models/sherpa-onnx-funasr-nano-int8-2025-12-30 \
  -e WYOMING_AUTO_DOWNLOAD=true \
  wyoming-sherpa-funasr:latest
```

### 3.4 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `WYOMING_HOST` | 监听地址 | `0.0.0.0` |
| `WYOMING_PORT` | 监听端口 | `10300` |
| `WYOMING_SERVICE_NAME` | mDNS 服务名称 | `sherpa-funasr` |
| `WYOMING_ZEROCONF` | 启用 Zeroconf | `true` |
| `WYOMING_MODEL_NAME` | 模型名称 | `sherpa-onnx-funasr-nano-int8-2025-12-30` |
| `WYOMING_MODEL_DIR` | 容器内模型目录 | `/data/models/sherpa-onnx-funasr-nano-int8-2025-12-30` |
| `WYOMING_SAMPLE_RATE` | 采样率 | `16000` |
| `WYOMING_NUM_THREADS` | 线程数 | `2` |
| `WYOMING_AUTO_DOWNLOAD` | 自动下载模型 | `true` |
| `WYOMING_DEBUG` | 调试日志 | `false` |
| `HOST_MODEL_DIR` | 宿主机模型目录 | `/data/models` |
| `CONTAINER_NAME` | 容器名称 | `wyoming-sherpa-funasr` |
| `IMAGE_NAME` | 镜像名称 | `wyoming-sherpa-funasr:latest` |

### 3.5 模型文件说明

- **模型持久化**：模型文件存储在宿主机的 `/data/models` 目录，删除容器不会丢失
- **自动下载**：首次启动时会自动下载模型（约 1GB），也可手动下载
- **手动下载**：
  ```bash
  cd /data
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  tar -xjf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
  ```

## 4. Python 虚拟环境部署

```bash
cd /wyoming-sherpa-funasr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

启动服务：

```bash
python run_server.py \
  --host 0.0.0.0 \
  --port 10300 \
  --service-name sherpa-funasr \
  --model-dir /data/models/sherpa-onnx-funasr-nano-int8-2025-12-30
```

若你不想广播 mDNS：

```bash
python run_server.py --model-dir /data/models/sherpa-onnx-funasr-nano-int8-2025-12-30 --no-zeroconf
```

## 5. 准备模型文件

从 sherpa-onnx 发布页下载 `sherpa-onnx-funasr-nano-int8-2025-12-30`，然后整理成如下结构（示例）：

```text
/data/models/sherpa-onnx-funasr-nano-int8-2025-12-30/
  encoder_adaptor.int8.onnx
  llm.int8.onnx
  embedding.int8.onnx
  Qwen3-0.6B/
    tokenizer.json
    vocab.json
    merges.txt
```

> 说明：不同版本的 `sherpa-onnx` Python API 参数名可能略有差异，本项目在运行时会动态适配 `from_funasr_nano()` 的参数。

## 6. 在虚拟环境内进行协议级测试

先安装测试依赖：

```bash
pip install soundfile
```

新建 `tests/manual_wyoming_client.py`：

```python
import json
import socket
import soundfile as sf
import numpy as np


def send(sock, typ, data=None, payload=b""):
    data = data or {}
    header = {"type": typ, "data": data}
    if payload:
        header["payload_length"] = len(payload)
    sock.sendall((json.dumps(header, ensure_ascii=True) + "\n").encode("utf-8"))
    if payload:
        sock.sendall(payload)


def recv_line(sock):
    buf = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise EOFError
        if b == b"\n":
            return bytes(buf)
        buf.extend(b)


def recv_msg(sock):
    header = json.loads(recv_line(sock).decode("utf-8"))
    payload = b""
    if header.get("payload_length"):
        n = int(header["payload_length"])
        while len(payload) < n:
            payload += sock.recv(n - len(payload))
    return header, payload


if __name__ == "__main__":
    wav, sr = sf.read("sample.wav", dtype="float32", always_2d=True)
    mono = wav.mean(axis=1)
    pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16).tobytes()

    with socket.create_connection(("127.0.0.1", 10300), timeout=10) as s:
        send(s, "describe")
        print("describe =>", recv_msg(s)[0])

        send(s, "transcribe", {"language": "zh"})
        send(s, "audio-start", {"rate": sr, "width": 2, "channels": 1})

        chunk = 3200  # 100ms @16k int16
        for i in range(0, len(pcm), chunk):
            send(s, "audio-chunk", {"rate": sr, "width": 2, "channels": 1}, pcm[i : i + chunk])

        send(s, "audio-stop")
        print("transcript =>", recv_msg(s)[0])
```

执行：

```bash
python tests/manual_wyoming_client.py
```

看到 `type=transcript` 且 `data.text` 有识别文本，说明服务可用。

## 7. Home Assistant 自发现与接入

1. 确保 HA 与该服务端在同一二层网络，且 mDNS 可达（UDP 5353 未被防火墙阻断）
2. 启动本服务（默认开启 zeroconf）
3. 在 HA 中进入 **设置 -> 设备与服务**
4. 正常情况下会自动弹出 Wyoming 发现项，选择并提交
5. 若没自动发现，手动添加 **Wyoming Protocol**，填服务器 IP + 端口（默认 `10300`）

## 8. 常见问题

- `from_funasr_nano` 报参数错误：
  - 升级 `sherpa-onnx` 到较新版本，或检查模型目录下文件是否齐全
- HA 无法自动发现：
  - 先确认服务正常监听端口
  - 检查 mDNS 跨网段是否被网关/交换机拦截
  - 可先用手动 host:port 方式验证
- Docker 容器无法启动：
  - 检查 `/data/models` 目录是否存在且有写入权限
  - 查看容器日志：`docker logs wyoming-sherpa-funasr`
  - 确保端口未被占用：`ss -tlnp | grep 10300`

## 9. 参考链接

- sherpa-onnx: <https://github.com/k2-fsa/sherpa-onnx>
- Wyoming 协议: <https://github.com/OHF-Voice/wyoming>
- FunASR Nano 文档: <https://k2-fsa.github.io/sherpa/onnx/funasr-nano/index.html#fun-asr-nano-2512>
