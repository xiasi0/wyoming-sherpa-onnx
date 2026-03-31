# Wyoming Qwen3-ASR Server

一个基于 Wyoming 协议的离线语音识别服务，只支持 `Qwen3-ASR`。

## 特性

- 仅支持 `Qwen3-ASR`
- 默认关闭 Home Assistant 自动发现
- 在 `audio-stop` 后返回最终 `transcript`
- 内置模型自动下载
- 单次音频固定限制为 `30s`

默认模型下载地址：

`https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2`

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_server.py
```

默认模型目录：

```text
/root/wyoming-sherpa-onnx/data/models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
```

如果目录下没有模型，服务会自动下载并解压。

查看调试日志：

```bash
python run_server.py --debug
```

## Docker

```bash
docker compose up -d --build
docker compose logs -f
```

容器内模型目录是：

```text
/data/models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
```

宿主机默认挂载目录是：

```text
${HOME}/models
```

## 模型目录结构

模型目录至少需要这些文件：

- `conv_frontend.onnx`
- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `tokenizer/vocab.json`
- `tokenizer/merges.txt`

## 手工测试

参考 [tests/manual_wyoming_client.py](/root/wyoming-sherpa-onnx/tests/manual_wyoming_client.py)。

## 说明

- 默认端口是 `10300`
- 默认 `ZEROCONF=false`
- 超过 `30s` 的音频会被丢弃，并在结束时返回空文本

## 参考项目

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Wyoming Protocol](https://github.com/OHF-Voice/wyoming)
