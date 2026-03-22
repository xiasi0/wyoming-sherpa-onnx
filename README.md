# Wyoming + sherpa-onnx ASR Server

基于 **Wyoming 协议** 的语音识别服务端，使用 `sherpa-onnx-funasr-nano-int8` 模型进行离线语音识别，支持通过 mDNS 广播供 Home Assistant 自动发现。

## 功能特性

- 支持 Wyoming 协议：`describe/info`、`transcribe`、`audio-start/chunk/stop`、`transcript`
- 支持 Zeroconf/mDNS 广播，Home Assistant 自动发现
- 支持多声道音频自动混合为单声道
- 支持非 16kHz 音频自动重采样到 16kHz
- 支持 Docker 部署，模型数据持久化
- 支持模型自动下载

## 目录结构

```text
wyoming-sherpa-funasr/
├── app/
│   ├── asr_engine.py      # FunASR Nano 引擎
│   ├── config.py          # 配置解析
│   ├── discovery.py       # mDNS 服务发现
│   ├── downloader.py      # 模型下载
│   ├── protocol.py        # Wyoming 协议
│   └── server.py          # Wyoming 服务器
├── data/                  # 模型数据目录
├── Dockerfile
├── docker-compose.yml
├── run_server.py
└── requirements.txt
```

## 快速开始

### Docker 部署（推荐）

#### 1. 使用 docker compose

```bash
cd wyoming-sherpa-funasr

# 构建并启动
docker compose up -d --build

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

#### 2. 使用 docker run

```bash
# 创建模型目录
mkdir -p /home/models

# 启动容器
docker run -d \
  --name wyoming-sherpa-onnx \
  --restart unless-stopped \
  -p 10300:10300 \
  -v /home/models:/data/models:rw \
  -e PORT=10300 \
  -e ZEROCONF=false \
  -e MODEL_DIR=/data/models/sherpa-onnx-funasr-nano-int8-2025-12-30 \
  -e AUTO_DOWNLOAD=true \
  wyoming-sherpa-onnx:latest
```

### 环境变量配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `HOST` | 监听地址 | `0.0.0.0` |
| `PORT` | 监听端口 | `10300` |
| `SERVICE_NAME` | mDNS 服务名称 | `wyoming-sherpa-onnx` |
| `ZEROCONF` | 启用 mDNS | `false` |
| `MODEL_NAME` | 模型名称 | `sherpa-onnx-funasr-nano-int8-2025-12-30` |
| `MODEL_DIR` | 模型目录 | `/data/models/sherpa-onnx-funasr-nano-int8-2025-12-30` |
| `SAMPLE_RATE` | 采样率 | `16000` |
| `NUM_THREADS` | 线程数 | `2` |
| `AUTO_DOWNLOAD` | 自动下载模型 | `true` |
| `DEBUG` | 调试日志 | `false` |

### 模型文件

模型存储在 `/home/models` 目录，结构如下：

```text
/home/models/sherpa-onnx-funasr-nano-int8-2025-12-30/
├── encoder_adaptor.int8.onnx
├── llm.int8.onnx
├── embedding.int8.onnx
└── Qwen3-0.6B/
    ├── tokenizer.json
    ├── vocab.json
    └── merges.txt
```

**手动下载模型**：

```bash
cd /home/models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
tar -xjf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
```

## Python 本地部署

```bash
cd wyoming-sherpa-funasr

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python run_server.py --model-dir /home/models/sherpa-onnx-funasr-nano-int8-2025-12-30
```

**禁用 mDNS**：

```bash
python run_server.py --model-dir /home/models/sherpa-onnx-funasr-nano-int8-2025-12-30 --no-zeroconf
```

## Home Assistant 接入

1. 确保 HA 与服务器在同一网络，mDNS 可达（UDP 5353 未被阻断）
2. 启动服务（`ZEROCONF=true`）
3. HA **设置 -> 设备与服务** 应自动发现 Wyoming 服务
4. 或手动添加 **Wyoming Protocol**，填写服务器 IP 和端口（默认 `10300`）

## 测试

安装测试依赖：

```bash
pip install soundfile
```

创建测试脚本 `tests/test_wyoming.py`：

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
        if not b or b == b"\n":
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
    # 读取 WAV 文件
    wav, sr = sf.read("sample.wav", dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes()

    with socket.create_connection(("127.0.0.1", 10300), timeout=10) as s:
        # 获取服务信息
        send(s, "describe")
        print("Info:", recv_msg(s)[0])

        # 发送音频进行识别
        send(s, "transcribe", {"language": "zh"})
        send(s, "audio-start", {"rate": sr, "width": 2, "channels": 1})

        chunk = 3200  # 100ms @16k int16
        for i in range(0, len(pcm), chunk):
            send(s, "audio-chunk", {}, pcm[i : i + chunk])

        send(s, "audio-stop")
        print("Transcript:", recv_msg(s)[0])
```

执行测试：

```bash
python tests/test_wyoming.py
```

## 常见问题

### 模型下载失败

检查网络连接，或手动下载模型：

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
```

### HA 无法自动发现

- 确认服务正常监听：`ss -tlnp | grep 10300`
- 检查 mDNS 是否可达（UDP 5353）
- 尝试手动添加 Wyoming 集成

### Docker 容器启动失败

```bash
# 查看日志
docker logs wyoming-sherpa-onnx

# 检查目录权限
ls -la /home/models
```

## 参考链接

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Wyoming 协议](https://github.com/OHF-Voice/wyoming)
- [FunASR Nano 文档](https://k2-fsa.github.io/sherpa/onnx/funasr-nano/index.html)
