# Wyoming Qwen3-ASR Server

一个基于 Wyoming 协议的离线 ASR 服务，仅支持 `Qwen3-ASR`。

## 快速开始

```bash
git clone <YOUR_REPO_URL> wyoming-sherpa-onnx
cd wyoming-sherpa-onnx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_server.py
```

调试日志：

```bash
python run_server.py --debug
```

## 模型规则

- 默认模型名：`sherpa-onnx-qwen3-asr-0.6B-int8`
- 可选模型名：`sherpa-onnx-qwen3-asr-1.7B-int8`
- 不需要手动传 `--model-dir`，会按 `--model-name` 自动映射目录
- 首次运行若本地无模型，会自动从 ModelScope 下载必要文件

ModelScope:
`https://www.modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx/files`

使用 1.7B：

```bash
python run_server.py --model-name sherpa-onnx-qwen3-asr-1.7B-int8
```

默认下载目录（项目根目录下）：

```text
data/models/
```

## Docker

```bash
git clone <YOUR_REPO_URL> wyoming-sherpa-onnx
cd wyoming-sherpa-onnx
docker compose up -d --build
docker compose logs -f
```

- 容器内模型目录：`/data/models`
- 宿主机挂载目录：`${HOME}/models`

## 模型必要文件

- `conv_frontend.onnx`
- `encoder.int8.onnx`
- `decoder.int8.onnx`
- `tokenizer/vocab.json`
- `tokenizer/merges.txt`
- `tokenizer/tokenizer_config.json`

## 说明

- 默认端口：`10300`
- 默认 `ZEROCONF=false`
- 单次音频上限：`30s`（超限返回空文本）

## 参考

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Wyoming Protocol](https://github.com/OHF-Voice/wyoming)
