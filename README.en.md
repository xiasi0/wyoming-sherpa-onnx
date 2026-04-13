# Wyoming Qwen3-ASR Server

Offline ASR server based on the Wyoming protocol, currently focused on Qwen3-ASR, with fixed-window speaker gating and GTCRN denoise.

## Highlights

- Qwen3-ASR only (`0.6B` / `1.7B`)
- Speaker gate disabled by default (multi-speaker directory enrollment, enable when needed)
- GTCRN denoise enabled by default
- Auto-download for missing models (ASR from ModelScope, speaker/denoise from sherpa-onnx releases)
- 30-second max audio per request

## Processing Pipeline

`Audio -> (optional) GTCRN denoise -> fixed-window speaker gate -> Qwen3-ASR`

## Known Limitations and Context

1. We still need a stronger algorithm for gate-passed segment reconstruction
- Current behavior: with fixed-window gating, middle segments can be rejected while head/tail segments pass.
- Gap: accurately reconstructing passed segments into a stable ASR input (without word drops/splits) is still an open problem in this project.
- Status: community contributions are welcome for better temporal smoothing/alignment after speaker gating.

2. HA currently cannot be force-stopped by STT server during capture
- Current behavior: even when many segments are rejected, HA voice assistant may keep listening until session ends.
- You may see many rejected segments in server logs while microphone capture continues.
- This is a HA/Wyoming client capability boundary and needs HA-side protocol/runtime improvements.

## Quick Start (Python)

```bash
git clone https://github.com/xiasi0/wyoming-sherpa-onnx.git
cd wyoming-sherpa-onnx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_server.py
```

Common commands:

```bash
python run_server.py --debug
python run_server.py --model-name sherpa-onnx-qwen3-asr-1.7B-int8
python run_server.py --hotwords "kids-room,light"
```

## Quick Start (Docker)

```bash
git clone https://github.com/xiasi0/wyoming-sherpa-onnx.git
cd wyoming-sherpa-onnx
docker compose up -d --build
docker compose logs -f
```

Default mounts:
- `${HOME}/data/models -> /app/data/models`
- `${HOME}/data/speaker_refs -> /data/speaker_refs`

## Model and Directory Rules

### 1) ASR Models

- Default: `sherpa-onnx-qwen3-asr-0.6B-int8`
- Optional: `sherpa-onnx-qwen3-asr-1.7B-int8`
- Select with `MODEL_NAME` (or `--model-name`)
- Auto-mapped directories:
  - `data/models/sherpa-onnx-qwen3-asr-0.6B-int8`
  - `data/models/sherpa-onnx-qwen3-asr-1.7B-int8`

Default model note:
- `0.6B` is recommended for daily HA short commands (lower resource usage).
- `1.7B` usually gives a higher ceiling but requires more CPU/memory.

### 2) Speaker Models

Select with `SPEAKER_MODEL_NAME` (or `--speaker-model-name`):
- `wespeaker-zh-cnceleb-resnet34` (Chinese, default)
- `wespeaker-en-voxceleb-resnet34` (English)
- `3dspeaker-campplus-zh-en` (Chinese/English multilingual)

Notes:
- `SPEAKER_MODEL_FILE` is an optional override.
- Default speaker model directory is fixed: `data/models/speaker`

### 3) Denoise Models

Select with `DENOISE_MODEL_NAME` (or `--denoise-model-name`):
- `gtcrn-simple` (default)

Notes:
- `DENOISE_MODEL_FILE` is an optional override.
- Default denoise model directory is fixed: `data/models/denoise`

## Multi-Speaker Enrollment

Required layout:

```text
data/speaker_refs/<speaker_id>/*.wav
```

Example:

```text
data/speaker_refs/alice/a1.wav
data/speaker_refs/alice/a2.wav
data/speaker_refs/bob/b1.wav
```

Important:
- Root-level `*.wav` directly under `data/speaker_refs` are ignored.
- Startup logs will warn when such files are detected.

## Default Configuration

- `SPEAKER_GATE=false`
- `SPEAKER_THRESHOLD=0.40`
- `DENOISE=true`
- `HOTWORDS=""`
- `ZEROCONF=false`

## CLI and Environment Quick Reference

| Feature | CLI | Env | Default |
|---|---|---|---|
| Host | `--host` | `HOST` | `0.0.0.0` |
| Port | `--port` | `PORT` | `10300` |
| ASR model | `--model-name` | `MODEL_NAME` | `sherpa-onnx-qwen3-asr-0.6B-int8` |
| Auto download | `--auto-download/--no-auto-download` | `AUTO_DOWNLOAD` | `true` |
| Debug logs | `--debug` | `DEBUG` | `false` |
| Hotwords | `--hotwords` | `HOTWORDS` | empty |
| Speaker gate | `--speaker-gate/--no-speaker-gate` | `SPEAKER_GATE` | `false` |
| Speaker threshold | `--speaker-threshold` | `SPEAKER_THRESHOLD` | `0.40` |
| Speaker model name | `--speaker-model-name` | `SPEAKER_MODEL_NAME` | `wespeaker-zh-cnceleb-resnet34` |
| Speaker model file override | `--speaker-model-file` | `SPEAKER_MODEL_FILE` | empty |
| Speaker model dir override | `--speaker-model-dir` | `SPEAKER_MODEL_DIR` | `data/models/speaker` |
| Speaker refs root | `--speaker-reference-dir` | `SPEAKER_REFERENCE_DIR` | `data/speaker_refs` |
| Speaker refs file list override | `--speaker-reference-wavs` | `SPEAKER_REFERENCE_WAVS` | empty (auto scan) |
| Denoise | `--denoise/--no-denoise` | `DENOISE` | `true` |
| Denoise model name | `--denoise-model-name` | `DENOISE_MODEL_NAME` | `gtcrn-simple` |
| Denoise model file override | `--denoise-model-file` | `DENOISE_MODEL_FILE` | empty |
| Denoise model dir override | N/A | `DENOISE_MODEL_DIR` | `data/models/denoise` |

## Upstream References

- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- Wyoming Protocol: https://github.com/OHF-Voice/wyoming
- Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR
- ModelScope Qwen3-ASR ONNX: https://www.modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx/files
- WeSpeaker: https://github.com/wenet-e2e/wespeaker
- 3D-Speaker: https://github.com/modelscope/3D-Speaker
