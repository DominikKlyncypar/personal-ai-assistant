from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..state import State
from ..config import Settings
from . import capture as cap_svc
from . import transcriber as asr_svc


def _resample_mono_f32(x: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    if x.size == 0 or src_hz == dst_hz:
        return x.astype(np.float32, copy=False)
    t_src = np.arange(x.shape[0], dtype=np.float32) / float(src_hz)
    n_dst = int(round(x.shape[0] * (dst_hz / float(src_hz))))
    if n_dst <= 1:
        return np.zeros(0, dtype=np.float32)
    t_dst = np.arange(n_dst, dtype=np.float32) / float(dst_hz)
    y = np.interp(t_dst, t_src, x).astype(np.float32)
    return y


def _f32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    if x.size == 0:
        return b""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def _take_latest_samples(state: State, seconds: float) -> Tuple[np.ndarray, int]:
    with state.capture.lock:
        sr = state.capture.samplerate
        blocks = list(state.capture.buffer)
    if not blocks:
        return np.zeros(0, dtype=np.float32), sr
    mono = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    n = int(seconds * sr)
    if mono.shape[0] > n:
        mono = mono[-n:]
    return mono.astype(np.float32, copy=False), sr


def _queue_auto_transcribe(
    state: State,
    seg_seconds: float,
    now: float,
    *,
    respect_window: bool = True,
    hard_limit: Optional[float] = None,
) -> bool:
    if seg_seconds <= 0:
        return False
    if not state.auto_transcribe:
        return False
    if state.auto_busy:
        return False
    if (now - state.last_trigger_ts) < state.auto_gap_s:
        return False

    max_secs = min(state.buffer_seconds, seg_seconds)
    if respect_window:
        max_secs = min(max_secs, state.auto_window_s)
    if hard_limit is not None:
        max_secs = min(max_secs, hard_limit)
    if max_secs <= 0:
        return False

    seg_len = max(1, int(round(max_secs)))
    state.auto_busy = True
    threading.Thread(target=_async_auto_transcribe, args=(state, seg_len), daemon=True).start()
    return True


def _async_auto_transcribe(state: State, seconds: int):
    try:
        resp = cap_svc.dump_wav(state, seconds=seconds, label="auto")
        if not resp.get("ok"):
            return
        p = resp.get("path")
        if not p:
            return
        res = asr_svc.transcribe_wav(
            state,
            path=str(p),
            language=None,
            beam_size=5,
            meeting_id=state.current_meeting_id,
        )
        if res.get("ok"):
            print(f">>> Auto-transcribed {resp.get('filename')}")
        else:
            print(f">>> Auto-transcribe error: {res}")
    except Exception as e:
        # Swallow async errors to avoid noisy thread exceptions
        try:
            import logging
            logging.getLogger("app").warning(f"auto_transcribe failed: {e}")
        except Exception:
            pass
    finally:
        state.last_trigger_ts = time.monotonic()
        state.auto_busy = False


def vad_check(state: State, settings: Settings, ng: float = -40.0, vad: int = 3) -> Dict[str, Any]:
    now = time.monotonic()
    recent, sr = _take_latest_samples(state, 0.1)
    running = bool(state.capture.running)
    if recent.size == 0:
        return {
            "ok": True,
            "speech": False,
            "rms": 0.0,
            "db": -120.0,
            "running": running,
            "speech_frames": int(state.vad.speech_frames),
            "silence_frames": int(state.vad.silence_frames),
            "ms_since_voice": int((now - (state.vad.last_voice_ts or now)) * 1000.0),
            "raw_is_speech_frame": False,
            "events": {"started": False, "ended": False, "failstop": False},
        }

    recent_16k = _resample_mono_f32(recent, sr, settings.vad_samplerate)
    frame_len = int(settings.vad_samplerate * (settings.vad_frame_ms / 1000.0))
    if recent_16k.shape[0] < frame_len:
        rms_short = float(np.sqrt(np.mean(np.square(recent_16k)))) if recent_16k.size else 0.0
        db_short = 20.0 * float(np.log10(max(rms_short, 1e-9)))
        return {
            "ok": True,
            "speech": False,
            "rms": rms_short,
            "db": db_short,
            "running": running,
            "speech_frames": int(state.vad.speech_frames),
            "silence_frames": int(state.vad.silence_frames),
            "ms_since_voice": int((now - (state.vad.last_voice_ts or now)) * 1000.0),
            "raw_is_speech_frame": False,
            "events": {"started": False, "ended": False, "failstop": False},
        }

    frame = recent_16k[-frame_len:]
    rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
    db_val = 20.0 * float(np.log10(max(rms, 1e-9)))

    if db_val < ng:
        is_speech_frame = False
    else:
        import webrtcvad  # lazy import

        vad_level = int(max(0, min(3, vad)))
        vad_inst = webrtcvad.Vad(vad_level)
        pcm_bytes = _f32_to_pcm16_bytes(frame)
        is_speech_frame = bool(pcm_bytes) and vad_inst.is_speech(pcm_bytes, settings.vad_samplerate)

    if is_speech_frame:
        state.vad.last_voice_ts = now

    if is_speech_frame:
        state.vad.speech_frames += 1
        state.vad.silence_frames = 0
    else:
        state.vad.silence_frames += 1
        state.vad.speech_frames = 0

    just_started = False
    just_ended = False
    failstop_triggered = False

    if not state.vad.speech_active and state.vad.speech_frames >= 2:
        state.vad.speech_active = True
        state.vad.segment_start_ts = now
        just_started = True
        state.vad.last_voice_ts = now

    if state.vad.speech_active:
        seg_start = state.vad.segment_start_ts or now
        segment_duration = now - seg_start

        if state.auto_failstop_s > 0 and segment_duration >= state.auto_failstop_s:
            if _queue_auto_transcribe(
                state,
                segment_duration,
                now,
                respect_window=False,
                hard_limit=state.auto_failstop_s,
            ):
                failstop_triggered = True
                overlap = max(
                    0.0,
                    min(
                        float(state.auto_failstop_overlap_s),
                        float(state.auto_failstop_s),
                        float(state.buffer_seconds),
                    ),
                )
                state.vad.segment_start_ts = now - overlap if overlap > 0 else now
                state.vad.last_voice_ts = now
                state.vad.speech_frames = 0
                state.vad.silence_frames = 0
                seg_start = state.vad.segment_start_ts or now
                segment_duration = max(0.0, now - seg_start)

        last_voice = state.vad.last_voice_ts or now
        ms_since_voice = (now - last_voice) * 1000.0
        if ms_since_voice >= settings.vad_hangover_ms:
            state.vad.speech_active = False
            just_ended = True
            seg_start = state.vad.segment_start_ts or seg_start
            segment_duration = now - seg_start
            _queue_auto_transcribe(state, segment_duration, now)
            state.vad.segment_start_ts = None
            state.vad.speech_frames = 0
            state.vad.silence_frames = 0

    return {
        "ok": True,
        "running": running,
        "rms": rms,
        "db": db_val,
        "speech": bool(state.vad.speech_active),
        "raw_is_speech_frame": bool(is_speech_frame),
        "speech_frames": int(state.vad.speech_frames),
        "silence_frames": int(state.vad.silence_frames),
        "ms_since_voice": int((now - (state.vad.last_voice_ts or now)) * 1000.0),
        "events": {"started": just_started, "ended": just_ended, "failstop": failstop_triggered},
    }
