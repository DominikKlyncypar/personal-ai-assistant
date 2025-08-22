# worker/src/main.py
from __future__ import annotations

import os
import time
import wave
import threading
from typing import Any, Dict, List
from collections import deque
import warnings

# Silence the pkg_resources deprecation warning emitted by webrtcvad import
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import sounddevice as sd
import webrtcvad
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Global state
# -----------------------------
CAPTURE_STATE: Dict[str, Any] = {"running": False}

VAD_SAMPLE_RATE = 16000   # valid: 8000, 16000, 32000, 48000
VAD_FRAME_MS = 30         # valid: 10, 20, 30

VAD_STATE = {
    "speech_active": False,
    "speech_frames": 0,
    "silence_frames": 0,
}

# -----------------------------
# Helpers (audio utils)
# -----------------------------
def _resample_mono_f32(x: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    """Linear resample mono float32 array from src_hz -> dst_hz."""
    if src_hz == dst_hz or x.size == 0:
        return x.astype(np.float32, copy=False)
    t_src = np.arange(x.shape[0], dtype=np.float32) / float(src_hz)
    n_dst = int(round(x.shape[0] * (dst_hz / float(src_hz))))
    if n_dst <= 1:
        return np.zeros(0, dtype=np.float32)
    t_dst = np.arange(n_dst, dtype=np.float32) / float(dst_hz)
    y = np.interp(t_dst, t_src, x).astype(np.float32)
    return y


def _f32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    """Clip float32 [-1,1] -> int16 PCM bytes."""
    if x.size == 0:
        return b""
    x = np.clip(x, -1.0, 1.0)
    pcm16 = (x * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def _safe_label(label: str) -> str:
    """Keep only letters, numbers, dash and underscore; collapse others to underscore."""
    import re

    label = (label or "").strip().lower()
    label = re.sub(r"[^a-z0-9_-]+", "_", label)
    return label or "snapshot"


def _write_wav(path: str, samplerate: int, mono_f32: np.ndarray) -> None:
    """Write mono float32 [-1,1] as 16-bit PCM WAV."""
    clipped = np.clip(mono_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(samplerate))
        wf.writeframes(pcm16.tobytes())


def _take_latest_samples(seconds: float) -> tuple[np.ndarray, int]:
    """Concatenate recent blocks and return last `seconds` of mono f32 + current samplerate."""
    with _CAPTURE.lock:
        sr = _CAPTURE.samplerate
        blocks = list(_CAPTURE.buffer)
    if not blocks:
        return np.zeros(0, dtype=np.float32), sr
    mono = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    n = int(seconds * sr)
    if mono.shape[0] > n:
        mono = mono[-n:]
    return mono.astype(np.float32, copy=False), sr


# -----------------------------
# Capture implementation
# -----------------------------
class _Capture:
    """
    Holds a sounddevice.InputStream and a rolling buffer of mono float32 audio.
    On Windows we can use WASAPI loopback for system playback capture.
    """
    def __init__(self):
        self.stream: sd.InputStream | None = None
        self.lock = threading.Lock()
        self.last_rms: float = 0.0
        self.running: bool = False
        self.samplerate: int = 48000
        # store ~30 seconds of ~20ms blocks: 30 / 0.02 = 1500
        self.buffer: deque[np.ndarray] = deque(maxlen=1500)

    def _start_stream(self, **kwargs):
        self.stream = sd.InputStream(**kwargs)
        self.stream.start()
        self.running = True

    def start_mic(self, device_id: int | None, samplerate: int):
        if self.running:
            return
        self.samplerate = samplerate
        blocksize = max(256, int(samplerate * 0.02))  # ~20ms
        channels = 1

        def callback(indata, frames, time_info, status):
            if status:
                # swallow glitches
                pass
            # Ensure mono float32 in [-1, 1]
            mono = indata[:, 0] if indata.ndim == 2 else indata
            mono = mono.astype(np.float32, copy=False)
            with self.lock:
                self.last_rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                # Copy to avoid reusing underlying buffer
                self.buffer.append(np.array(mono, dtype=np.float32))

        self._start_stream(
            device=device_id if device_id is not None else None,
            channels=channels,
            samplerate=samplerate,
            blocksize=blocksize,
            dtype="float32",
            callback=callback,
        )

    def start_playback_loopback(self, device_id: int | None, samplerate: int):
        """Windows-only: capture system playback (WASAPI loopback)."""
        if self.running:
            return
        import platform

        if platform.system() != "Windows":
            raise RuntimeError("Loopback capture requires Windows (WASAPI).")
        self.samplerate = samplerate
        blocksize = max(256, int(samplerate * 0.02))  # ~20ms
        channels = 2  # speakers usually stereo

        try:
            wasapi = sd.WasapiSettings(loopback=True)
        except Exception as e:
            raise RuntimeError(f"WASAPI loopback not available: {e}")

        def callback(indata, frames, time_info, status):
            if status:
                pass
            # stereo -> mono
            mono = np.mean(indata, axis=1, dtype=np.float32)
            with self.lock:
                self.last_rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                self.buffer.append(np.array(mono, dtype=np.float32))

        self._start_stream(
            device=device_id if device_id is not None else None,
            channels=channels,
            samplerate=samplerate,
            blocksize=blocksize,
            dtype="float32",
            callback=callback,
            extra_settings=wasapi,
        )

    def stop(self):
        if not self.running:
            return
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None
            self.running = False
            with self.lock:
                self.last_rms = 0.0
                # we keep buffer so /dump_wav can still write the last seconds if desired

    def get_level(self) -> float:
        with self.lock:
            return self.last_rms


_CAPTURE = _Capture()

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/devices")
def list_devices():
    """List input/output devices and host APIs."""
    info = sd.query_devices()
    hostapis = sd.query_hostapis()
    # Build lists
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    # Map hostapi index -> name
    api_names = {i: hostapis[i]["name"] for i in range(len(hostapis))}
    for idx, dev in enumerate(info):
        dev_api = api_names.get(dev["hostapi"], "unknown")
        item = {
            "id": idx,
            "name": str(dev["name"]),
            "hostapi": dev_api,
            "max_input_channels": int(dev.get("max_input_channels", 0)),
            "max_output_channels": int(dev.get("max_output_channels", 0)),
            "default_samplerate": float(dev.get("default_samplerate", 48000)),
        }
        if item["max_input_channels"] > 0:
            inputs.append(item)
        if item["max_output_channels"] > 0:
            # Only Windows WASAPI supports loopback natively through sounddevice
            import platform

            item_out = dict(item)
            item_out["loopback_capable"] = platform.system() == "Windows"
            outputs.append(item_out)

    return {
        "os": os.name if os.name != "nt" else "Windows",
        "hostapis": [h["name"] for h in hostapis],
        "inputs": inputs,
        "outputs": outputs,
        "notes": "On Windows with WASAPI, choose an OUTPUT device for loopback capture. Mic is optional.",
    }


@app.post("/start_capture")
def start_capture(payload: Dict[str, Any]):
    """
    Start capture.
    payload: { "playback_id": int | null, "mic_id": int | null, "samplerate": int }
    On Windows: set playback_id to capture loopback. Mic is optional.
    On macOS: loopback is not available via sounddevice; use mic_id.
    """
    playback_id = payload.get("playback_id", None)
    mic_id = payload.get("mic_id", None)
    samplerate = int(payload.get("samplerate", 48000))

    if CAPTURE_STATE["running"]:
        return {"ok": True, "message": "already running", "running": True}

    # Prefer mic if provided; else try playback loopback (Windows only)
    try:
        if mic_id is not None and mic_id != "":
            _CAPTURE.start_mic(int(mic_id), samplerate)
        else:
            _CAPTURE.start_playback_loopback(int(playback_id) if playback_id is not None else None, samplerate)
        CAPTURE_STATE["running"] = True
        # reset hysteresis counters
        VAD_STATE["speech_active"] = False
        VAD_STATE["speech_frames"] = 0
        VAD_STATE["silence_frames"] = 0
        return {"ok": True, "message": "started", "running": True, "playback_id": playback_id, "mic_id": mic_id, "samplerate": samplerate}
    except Exception as e:
        CAPTURE_STATE["running"] = False
        _CAPTURE.stop()
        return {"ok": False, "message": str(e), "running": False}


@app.post("/stop_capture")
def stop_capture():
    _CAPTURE.stop()
    CAPTURE_STATE["running"] = False
    # reset hysteresis
    VAD_STATE["speech_active"] = False
    VAD_STATE["speech_frames"] = 0
    VAD_STATE["silence_frames"] = 0
    return {"ok": True, "message": "stopped", "running": False}


@app.get("/capture_status")
def capture_status():
    return {"running": bool(CAPTURE_STATE["running"])}


@app.get("/level")
def level():
    return {"running": bool(CAPTURE_STATE["running"]), "rms": float(_CAPTURE.get_level())}


@app.get("/dump_wav")
def dump_wav(seconds: int = 5, label: str | None = None):
    """
    Write the last `seconds` of audio to worker/tmp/<label>_<timestamp>.wav and return the path.
    """
    if seconds <= 0 or seconds > 30:
        seconds = 5

    # gather recent blocks
    with _CAPTURE.lock:
        samplerate = _CAPTURE.samplerate
        blocks = list(_CAPTURE.buffer)

    if not blocks:
        return {"ok": False, "message": "no audio in buffer"}

    buf = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    # take the last N seconds
    take = int(seconds * samplerate)
    if buf.shape[0] > take:
        buf = buf[-take:]

    # ensure output dir
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp"))
    os.makedirs(out_dir, exist_ok=True)

    # unique name: <label>_YYYYmmdd-HHMMSS.wav (append _2, _3 if needed)
    lbl = _safe_label(label) if label else "snapshot"
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"{lbl}_{ts}"
    filename = base + ".wav"
    out_path = os.path.join(out_dir, filename)
    i = 2
    while os.path.exists(out_path):
        filename = f"{base}_{i}.wav"
        out_path = os.path.join(out_dir, filename)
        i += 1

    _write_wav(out_path, samplerate, buf)
    return {
        "ok": True,
        "path": os.path.abspath(out_path),
        "filename": filename,
        "seconds": int(seconds),
        "samplerate": int(samplerate),
        "samples": int(buf.shape[0]),
    }


@app.get("/vad_check")
def vad_check(ng: float = -40.0, vad: int = 3):
    """
    VAD with noise gate & hysteresis.
    Query params:
      - ng (float): Noise gate in dB (e.g., -40)
      - vad (int): WebRTC VAD aggressiveness 0..3
    Returns:
      { ok, speech, rms, db, running }
    """
    recent, sr = _take_latest_samples(0.1)  # ~100 ms of audio
    running = bool(CAPTURE_STATE["running"])

    if recent.size == 0:
        return {"ok": True, "speech": False, "rms": 0.0, "db": -120.0, "running": running}

    # resample to VAD rate; take the last 30 ms frame
    recent_16k = _resample_mono_f32(recent, sr, VAD_SAMPLE_RATE)
    frame_len = int(VAD_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))  # e.g., 480 @ 16k, 30ms
    if recent_16k.shape[0] < frame_len:
        rms_short = float(np.sqrt(np.mean(np.square(recent_16k)))) if recent_16k.size else 0.0
        db_short = 20.0 * float(np.log10(max(rms_short, 1e-9)))
        return {"ok": True, "speech": False, "rms": rms_short, "db": db_short, "running": running}

    frame = recent_16k[-frame_len:]
    rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
    db = 20.0 * float(np.log10(max(rms, 1e-9)))

    # noise gate first
    if db < ng:
        is_speech_frame = False
    else:
        # clamp vad param
        vad_level = int(vad)
        if vad_level < 0:
            vad_level = 0
        if vad_level > 3:
            vad_level = 3
        vad_inst = webrtcvad.Vad(vad_level)
        pcm_bytes = _f32_to_pcm16_bytes(frame)
        is_speech_frame = bool(pcm_bytes) and vad_inst.is_speech(pcm_bytes, VAD_SAMPLE_RATE)

    # hysteresis update
    if is_speech_frame:
        VAD_STATE["speech_frames"] += 1
        VAD_STATE["silence_frames"] = 0
    else:
        VAD_STATE["silence_frames"] += 1
        VAD_STATE["speech_frames"] = 0

    # thresholds (tunable)
    start_thresh = 4   # ≈120 ms of speech to start
    end_thresh = 8     # ≈240 ms of silence to end

    if not VAD_STATE["speech_active"] and VAD_STATE["speech_frames"] >= start_thresh:
        VAD_STATE["speech_active"] = True
        print(">>> Speech started")

    if VAD_STATE["speech_active"] and VAD_STATE["silence_frames"] >= end_thresh:
        VAD_STATE["speech_active"] = False
        print(">>> Speech ended")

    return {
        "ok": True,
        "speech": bool(VAD_STATE["speech_active"]),
        "rms": rms,
        "db": db,
        "running": running,
    }