from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .asr_engine import AudioFormat, Qwen3AsrEngine
from .config import AppConfig
from .denoise import GtcrnEnhancer
from .protocol import read_message, write_message
from .speaker_gate import SpeakerGate

LOGGER = logging.getLogger("wyoming-sherpa-onnx")
_MAX_AUDIO_SECONDS = 30.0
_SPEAKER_GATE_WINDOW_SECONDS = 0.8


def _is_disconnect_error(exc: BaseException) -> bool:
    return isinstance(exc, (BrokenPipeError, ConnectionResetError))


@dataclass(slots=True)
class SessionState:
    transcribe_opts: dict[str, Any]
    audio_format: AudioFormat | None
    stream: Any | None
    chunk_count: int
    total_bytes: int
    over_limit: bool
    detected_segments: int
    accepted_segments: int
    rejected_segments: int
    accepted_samples: int
    gate_pending_waveform: np.ndarray
    gate_pending_start_idx: int


class WyomingAsrServer:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.engine = Qwen3AsrEngine(
            model_dir=cfg.model_dir,
            sample_rate=cfg.sample_rate,
            num_threads=cfg.num_threads,
            hotwords=cfg.hotwords,
        )
        self.speaker_gate: SpeakerGate | None = None
        if cfg.speaker_gate:
            self.speaker_gate = SpeakerGate(
                model_path=cfg.speaker_model_dir / cfg.speaker_model_file,
                reference_wavs=cfg.speaker_reference_wavs,
                threshold=cfg.speaker_threshold,
                num_threads=max(1, cfg.num_threads),
                reference_root=cfg.speaker_reference_dir,
            )
        self.denoiser: GtcrnEnhancer | None = None
        if cfg.denoise_enabled:
            self.denoiser = GtcrnEnhancer(
                model_path=cfg.denoise_model_dir / cfg.denoise_model_file,
                num_threads=max(1, cfg.num_threads),
            )
        self._info_cache: dict[str, Any] | None = None
        self._server: asyncio.AbstractServer | None = None

    async def handle_client(self, reader, writer) -> None:
        peer = self._get_peername(writer)
        LOGGER.info("Client connected: %s", peer)
        state = SessionState(
            transcribe_opts={},
            audio_format=None,
            stream=None,
            chunk_count=0,
            total_bytes=0,
            over_limit=False,
            detected_segments=0,
            accepted_segments=0,
            rejected_segments=0,
            accepted_samples=0,
            gate_pending_waveform=np.empty((0,), dtype=np.float32),
            gate_pending_start_idx=0,
        )

        try:
            while True:
                msg = await read_message(reader)
                data = {**msg.data, **msg.extra_data}
                LOGGER.debug(
                    "[%s] Received message type=%s data_keys=%s payload_bytes=%s",
                    peer,
                    msg.msg_type,
                    sorted(data.keys()),
                    len(msg.payload),
                )

                if msg.msg_type == "describe":
                    await write_message(writer, "info", self._get_info())
                elif msg.msg_type == "transcribe":
                    state.transcribe_opts = data
                elif msg.msg_type == "audio-start":
                    state.stream = self.engine.create_stream()
                    state.audio_format = AudioFormat(
                        rate=int(data.get("rate", self.cfg.sample_rate)),
                        width=int(data.get("width", 2)),
                        channels=int(data.get("channels", 1)),
                    )
                    state.chunk_count = 0
                    state.total_bytes = 0
                    state.over_limit = False
                    state.detected_segments = 0
                    state.accepted_segments = 0
                    state.rejected_segments = 0
                    state.accepted_samples = 0
                    state.gate_pending_waveform = np.empty((0,), dtype=np.float32)
                    state.gate_pending_start_idx = 0
                    if self.denoiser is not None:
                        self.denoiser.reset()
                    LOGGER.debug(
                        "[%s] Audio stream started: %dHz, %d-bit, %d channels",
                        peer,
                        state.audio_format.rate,
                        state.audio_format.width * 8,
                        state.audio_format.channels,
                    )
                elif msg.msg_type == "audio-chunk":
                    if msg.payload and state.stream is not None and state.audio_format is not None:
                        bytes_per_sample = state.audio_format.width * state.audio_format.channels
                        total_bytes = state.total_bytes + len(msg.payload)
                        audio_duration = total_bytes / (bytes_per_sample * state.audio_format.rate)
                        if audio_duration > _MAX_AUDIO_SECONDS:
                            state.over_limit = True
                            state.stream = None
                            LOGGER.warning(
                                "[%s] Audio exceeded max duration %.2fs, dropping request at %.2fs",
                                peer,
                                _MAX_AUDIO_SECONDS,
                                audio_duration,
                            )
                            continue
                        waveform = self.engine.pcm_chunk_to_model_waveform(
                            msg.payload, state.audio_format
                        )
                        if self.denoiser is not None:
                            waveform = self.denoiser.enhance(waveform, self.cfg.sample_rate)
                        self._process_segments(peer, state, [waveform])
                        state.chunk_count += 1
                        state.total_bytes += len(msg.payload)
                        LOGGER.debug(
                            "[%s] Audio chunk #%d received (%.1f KB)",
                            peer,
                            state.chunk_count,
                            len(msg.payload) / 1024,
                        )
                elif msg.msg_type == "audio-stop":
                    text = ""
                    if state.over_limit:
                        LOGGER.warning(
                            "[%s] Returning empty transcript because audio exceeded %.2fs limit",
                            peer,
                            _MAX_AUDIO_SECONDS,
                        )
                    elif state.stream is not None:
                        audio_duration = 0.0
                        if state.audio_format and state.total_bytes > 0:
                            bytes_per_sample = state.audio_format.width * state.audio_format.channels
                            audio_duration = state.total_bytes / (bytes_per_sample * state.audio_format.rate)

                        if self.denoiser is not None:
                            tail = self.denoiser.flush()
                            if tail.size > 0:
                                self._process_segments(peer, state, [tail])
                        self._flush_pending_speaker_gate(peer, state, force=True)

                        start_time = time.time()
                        text = self.engine.finish_stream(state.stream)
                        infer_time = time.time() - start_time
                        rtf = infer_time / audio_duration if audio_duration > 0 else 0.0

                        state.stream = None

                        LOGGER.info(
                            "[%s] Recognition completed: audio=%.2fs, inference=%.3fs, RTF=%.2f, "
                            "segments=%d, speaker_accept=%d, speaker_reject=%d, asr_fed=%.2fs, result=\"%s\"",
                            peer,
                            audio_duration,
                            infer_time,
                            rtf,
                            state.detected_segments,
                            state.accepted_segments,
                            state.rejected_segments,
                            state.accepted_samples / float(self.cfg.sample_rate),
                            text if text else "(silence)",
                        )
                        if state.accepted_segments == 0 and state.rejected_segments > 0:
                            LOGGER.info(
                                "[%s] Empty transcript because all detected speech segments "
                                "were rejected by speaker gate.",
                                peer,
                            )
                        elif state.accepted_segments == 0:
                            LOGGER.info(
                                "[%s] Empty transcript because no audio segments passed to ASR.",
                                peer,
                            )

                    await write_message(
                        writer,
                        "transcript",
                        {
                            "text": text,
                            "language": state.transcribe_opts.get("language", "zh"),
                        },
                    )
                    LOGGER.debug("[%s] Sent transcript: %r", peer, text)
                    state.over_limit = False
                    state.gate_pending_waveform = np.empty((0,), dtype=np.float32)
                    state.gate_pending_start_idx = 0
                else:
                    LOGGER.debug("Ignoring unsupported message type: %s", msg.msg_type)
        except EOFError:
            LOGGER.info("Client disconnected: %s", peer)
        except (BrokenPipeError, ConnectionResetError) as exc:
            LOGGER.info("Client connection closed while streaming: %s (%s)", peer, exc)
        except asyncio.CancelledError:
            LOGGER.info("Client session cancelled: %s", peer)
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Client session error: %s", exc)
        finally:
            try:
                writer.close()
            except Exception:
                pass

            try:
                await writer.wait_closed()
            except Exception as exc:  # noqa: BLE001
                if not _is_disconnect_error(exc):
                    LOGGER.debug("Error while closing client stream %s: %s", peer, exc)

    def _get_peername(self, writer) -> str:
        """获取客户端地址，安全处理异常。"""
        try:
            peer = writer.get_extra_info("peername")
            return str(peer) if peer else "unknown"
        except Exception:
            return "unknown"

    def _get_info(self) -> dict[str, Any]:
        """获取 Wyoming 协议 info 响应（带缓存）。"""
        if self._info_cache is None:
            self._info_cache = {
                "asr": [
                    {
                        "name": self.cfg.model_name,
                        "attribution": {
                            "name": "k2-fsa/sherpa-onnx",
                            "url": "https://github.com/k2-fsa/sherpa-onnx",
                        },
                        "installed": True,
                        "models": [
                            {
                                "name": self.cfg.model_name,
                                "languages": ["zh", "en"],
                                "attribution": {
                                    "name": "k2-fsa/sherpa-onnx",
                                    "url": "https://github.com/k2-fsa/sherpa-onnx",
                                },
                                "installed": True,
                                "description": "Qwen3-ASR via sherpa-onnx",
                            }
                        ],
                        "supports_transcript_streaming": False,
                    }
                ],
            }
        return self._info_cache

    def _process_segments(
        self,
        peer: str,
        state: SessionState,
        segments: list,
    ) -> None:
        if state.stream is None:
            return
        for segment in segments:
            state.detected_segments += 1
            if self.speaker_gate is None:
                self.engine.feed_waveform_to_stream(state.stream, segment)
                state.accepted_segments += 1
                state.accepted_samples += len(segment)
                seg_sec = len(segment) / float(self.cfg.sample_rate)
                LOGGER.info(
                    "[%s] Segment #%d fed to ASR: dur=%.2fs samples=%d accepted=%d rejected=%d asr_fed=%.2fs",
                    peer,
                    state.detected_segments,
                    seg_sec,
                    len(segment),
                    state.accepted_segments,
                    state.rejected_segments,
                    state.accepted_samples / float(self.cfg.sample_rate),
                )
                continue

            if state.gate_pending_waveform.size == 0:
                state.gate_pending_start_idx = state.detected_segments
                state.gate_pending_waveform = np.ascontiguousarray(segment, dtype=np.float32)
            else:
                state.gate_pending_waveform = np.concatenate(
                    (state.gate_pending_waveform, segment)
                )
            self._flush_pending_speaker_gate(peer, state, force=False)

    def _flush_pending_speaker_gate(
        self, peer: str, state: SessionState, force: bool
    ) -> None:
        if self.speaker_gate is None:
            return
        if state.stream is None:
            return
        if state.gate_pending_waveform.size == 0:
            return

        gate_window = max(1, int(self.cfg.sample_rate * _SPEAKER_GATE_WINDOW_SECONDS))

        while state.gate_pending_waveform.size >= gate_window:
            seg = np.ascontiguousarray(state.gate_pending_waveform[:gate_window], dtype=np.float32)
            self._eval_gate_segment(peer, state, seg, float(self.cfg.speaker_threshold))
            state.gate_pending_waveform = state.gate_pending_waveform[gate_window:]
            state.gate_pending_start_idx = state.detected_segments

        if force and state.gate_pending_waveform.size > 0:
            seg = np.ascontiguousarray(state.gate_pending_waveform, dtype=np.float32)
            seg_sec = seg.size / float(self.cfg.sample_rate)
            effective_threshold = float(self.cfg.speaker_threshold)
            if seg_sec < _SPEAKER_GATE_WINDOW_SECONDS:
                # Tail segment is shorter than fixed window; slightly relax threshold.
                relax = (1.0 - max(0.0, seg_sec / _SPEAKER_GATE_WINDOW_SECONDS)) * 0.10
                effective_threshold = max(0.20, effective_threshold - relax)
            self._eval_gate_segment(peer, state, seg, effective_threshold)
            state.gate_pending_waveform = np.empty((0,), dtype=np.float32)
            state.gate_pending_start_idx = 0

    def _eval_gate_segment(
        self, peer: str, state: SessionState, segment: np.ndarray, threshold: float
    ) -> None:
        if self.speaker_gate is None or state.stream is None:
            return

        start_idx = state.gate_pending_start_idx or 1
        end_idx = state.detected_segments
        effective_threshold = float(threshold)
        if start_idx <= 1:
            # Relax first segment slightly to avoid missing weak wake-up/command onset.
            effective_threshold = max(0.20, effective_threshold - 0.05)

        accepted, similarity, speaker_id = self.speaker_gate.accepts_waveform(
            segment,
            self.cfg.sample_rate,
            threshold=effective_threshold,
        )
        samples = int(segment.size)
        seg_sec = samples / float(self.cfg.sample_rate)

        if not accepted:
            state.rejected_segments += 1
            LOGGER.info(
                "[%s] Fixed segment #%d-%d rejected: dur=%.2fs samples=%d similarity=%.3f threshold=%.3f "
                "matched=%s accepted=%d rejected=%d",
                peer,
                start_idx,
                end_idx,
                seg_sec,
                samples,
                similarity,
                effective_threshold,
                speaker_id or "-",
                state.accepted_segments,
                state.rejected_segments,
            )
            return

        self.engine.feed_waveform_to_stream(state.stream, segment)
        state.accepted_segments += 1
        state.accepted_samples += samples
        LOGGER.info(
            "[%s] Fixed segment #%d-%d fed to ASR: dur=%.2fs samples=%d similarity=%.3f threshold=%.3f "
            "matched=%s accepted=%d rejected=%d asr_fed=%.2fs",
            peer,
            start_idx,
            end_idx,
            seg_sec,
            samples,
            similarity,
            effective_threshold,
            speaker_id or "-",
            state.accepted_segments,
            state.rejected_segments,
            state.accepted_samples / float(self.cfg.sample_rate),
        )

    async def run(self) -> None:
        self._server = await asyncio.start_server(self.handle_client, self.cfg.host, self.cfg.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in self._server.sockets or [])
        LOGGER.info("Wyoming server listening on %s", addrs)
        LOGGER.info("Wyoming Qwen3-ASR service started successfully")
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is None:
            return

        self._server.close()
        await self._server.wait_closed()
        self._server = None
        LOGGER.info("Wyoming server stopped")
