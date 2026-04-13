# Wyoming Qwen3-ASR Server

基于 Wyoming 协议的离线 ASR 服务，当前仅支持 Qwen3-ASR，并内置固定窗口声纹门控与 GTCRN 降噪能力。

[English README](./README.en.md)

## 核心特性

- 仅支持 Qwen3-ASR（`0.6B` / `1.7B`）
- 默认关闭声纹门控（多说话人目录注册，可按需开启）
- 默认开启 GTCRN 降噪
- 自动下载缺失模型（ASR 来自 ModelScope，声纹/降噪来自 sherpa-onnx release）
- 单次音频上限 30 秒

## 处理链路

`Audio -> (可选)GTCRN降噪 -> 固定窗口声纹门控 -> Qwen3-ASR`

## 已知问题与背景说明

当前我们在真实家庭语音场景里遇到两类已知问题，先说明清楚，避免误判为配置错误：

1. 声纹门控后的“有效语音拼接”仍缺少成熟算法
- 现状：固定窗口门控会出现“中间段被拒绝、首尾段通过”的情况。
- 问题：如何把这些离散通过段做更精确的时序重建，并稳定送入 ASR（避免丢词/断词），目前项目内还没有足够成熟的算法实现。
- 结论：这一块需要社区进一步贡献更稳健的门控后处理算法（如更强的时序平滑/对齐策略）。

2. HA 侧暂不支持由 STT 服务主动停止麦克风采集
- 现状：即使服务端已经判定后续音频段被拒绝，HA 语音助手通常仍会继续收听到会话结束。
- 表现：日志中会看到大量 rejected 片段，但客户端麦克风不会被 STT 服务端直接停掉。
- 结论：这属于 HA/Wyoming 客户端能力边界，需要 HA 侧协议与实现支持“服务端触发停采”能力后才能根治。

## 快速开始（Python）

```bash
git clone https://github.com/xiasi0/wyoming-sherpa-onnx.git
cd wyoming-sherpa-onnx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_server.py
```

常用命令：

```bash
python run_server.py --debug
python run_server.py --model-name sherpa-onnx-qwen3-asr-1.7B-int8
python run_server.py --hotwords "儿童房,台灯"
```

## 快速开始（Docker）

```bash
git clone https://github.com/xiasi0/wyoming-sherpa-onnx.git
cd wyoming-sherpa-onnx
docker compose up -d --build
docker compose logs -f
```

默认挂载：
- `${HOME}/data -> /data`

## 模型与目录规则

### 1) ASR 模型

- 默认：`sherpa-onnx-qwen3-asr-0.6B-int8`
- 可选：`sherpa-onnx-qwen3-asr-1.7B-int8`
- 通过 `MODEL_NAME`（或 `--model-name`）切换
- 目录自动映射：
  - `data/models/sherpa-onnx-qwen3-asr-0.6B-int8`
  - `data/models/sherpa-onnx-qwen3-asr-1.7B-int8`

默认模型说明：
- `sherpa-onnx-qwen3-asr-0.6B-int8` 是默认推荐模型，资源占用更低，适合日常 HA 短指令。
- `1.7B` 适合你要追求更高识别上限的场景，但 CPU/内存开销更高。

### 2) 声纹模型

通过 `SPEAKER_MODEL_NAME`（或 `--speaker-model-name`）选择：
- `wespeaker-zh-cnceleb-resnet34`（中文，默认）
- `wespeaker-en-voxceleb-resnet34`（英文）
- `3dspeaker-campplus-zh-en`（中英多语）

说明：
- `SPEAKER_MODEL_FILE` 为可选覆盖项
- 声纹模型目录默认按 `MODEL_NAME` 推导：`data/models/<asr_model_dir>/speaker`

### 3) 降噪模型

通过 `DENOISE_MODEL_NAME`（或 `--denoise-model-name`）选择：
- `gtcrn-simple`（默认）

说明：
- `DENOISE_MODEL_FILE` 为可选覆盖项
- 降噪模型目录默认按 `MODEL_NAME` 推导：`data/models/<asr_model_dir>/denoise`

## 多说话人注册规则

注册目录必须是：

```text
data/speaker_refs/<speaker_id>/*.wav
```

示例：

```text
data/speaker_refs/alice/a1.wav
data/speaker_refs/alice/a2.wav
data/speaker_refs/bob/b1.wav
```

注意：
- `data/speaker_refs` 根目录下的 `*.wav` 会被忽略
- 启动时会给出忽略告警

## 默认配置

- `SPEAKER_GATE=false`
- `SPEAKER_THRESHOLD=0.40`
- `DENOISE=true`
- `HOTWORDS=""`
- `ZEROCONF=false`

## 下载与校验逻辑

ASR / 声纹 / 降噪三类模型采用统一逻辑：

- 启动时先校验目标模型文件是否存在且可用
- 存在则跳过下载（不会重复下载）
- 缺失时按当前配置的模型名/文件名自动下载
- 下载完成后再次校验，校验失败则启动失败并打印下载地址

## 参数与环境变量（速查）

| 功能 | CLI 参数 | 环境变量 | 默认值 |
|---|---|---|---|
| 监听地址 | `--host` | `HOST` | `0.0.0.0` |
| 监听端口 | `--port` | `PORT` | `10300` |
| 模型名 | `--model-name` | `MODEL_NAME` | `sherpa-onnx-qwen3-asr-0.6B-int8` |
| 自动下载 | `--auto-download/--no-auto-download` | `AUTO_DOWNLOAD` | `true` |
| 调试日志 | `--debug` | `DEBUG` | `false` |
| 热词 | `--hotwords` | `HOTWORDS` | 空 |
| 声纹门控开关 | `--speaker-gate/--no-speaker-gate` | `SPEAKER_GATE` | `false` |
| 声纹阈值 | `--speaker-threshold` | `SPEAKER_THRESHOLD` | `0.40` |
| 声纹模型名 | `--speaker-model-name` | `SPEAKER_MODEL_NAME` | `wespeaker-zh-cnceleb-resnet34` |
| 声纹模型文件覆盖 | `--speaker-model-file` | `SPEAKER_MODEL_FILE` | 空 |
| 声纹模型目录覆盖 | `--speaker-model-dir` | `SPEAKER_MODEL_DIR` | 自动按 `MODEL_NAME` 映射 |
| 声纹参考目录 | `--speaker-reference-dir` | `SPEAKER_REFERENCE_DIR` | `data/speaker_refs` |
| 声纹参考文件覆盖 | `--speaker-reference-wavs` | `SPEAKER_REFERENCE_WAVS` | 空（自动扫描目录） |
| 降噪开关 | `--denoise/--no-denoise` | `DENOISE` | `true` |
| 降噪模型名 | `--denoise-model-name` | `DENOISE_MODEL_NAME` | `gtcrn-simple` |
| 降噪模型文件覆盖 | `--denoise-model-file` | `DENOISE_MODEL_FILE` | 空 |
| 降噪模型目录覆盖 | 无 | `DENOISE_MODEL_DIR` | 自动按 `MODEL_NAME` 映射 |

## 上游引用

本项目直接引用或依赖的上游仓库/资源：

- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- Wyoming Protocol: https://github.com/OHF-Voice/wyoming
- Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR
- ModelScope Qwen3-ASR ONNX: https://www.modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx/files
- WeSpeaker: https://github.com/wenet-e2e/wespeaker
- 3D-Speaker: https://github.com/modelscope/3D-Speaker
