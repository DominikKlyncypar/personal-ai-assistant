# worker/src/main.py
from __future__ import annotations

import os
import re
import time
import wave
import threading
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import difflib
import numpy as np
import sounddevice as sd
import webrtcvad
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field

# Package-local DB helpers
from . import db
from .db import (
    initialize_db,
    new_meeting,
    delete_meeting,
    insert_utterance,
)

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


@app.on_event("startup")
def _init_db_on_startup():
    """
    Ensure DB file and tables exist.
    """
    initialize_db()


# -----------------------------
# Global state
# -----------------------------
CAPTURE_STATE: Dict[str, Any] = {"running": False}

# VAD config
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 20            # 20ms frames for WebRTC VAD
START_THRESH_FRAMES = 2      # ~40ms to declare "speech started"
HANGOVER_MS = 800            # keep speech active this long after last voiced frame

# Audio buffer/windowing
BUFFER_SECONDS = 180         # keep up to 3 minutes of audio in RAM
AUTO_MAX_WINDOW_S = 180      # cap any single auto-transcribed segment to 3 min

# Single source of truth used by UI & inserts
CURRENT_MEETING_ID: int | None = None

# VAD state
VAD_STATE: Dict[str, Any] = {
    "speech_active": False,
    "speech_frames": 0,
    "silence_frames": 0,
    "last_voice_ts": None,     # monotonic seconds (float)
    "segment_start_ts": None,  # monotonic seconds (float)
}

# In-memory transcripts (debug/legacy)
TRANSCRIPTS: List[Dict[str, Any]] = []
TRANSCRIPTS_LOCK = threading.Lock()
NEXT_UTTER_ID = 1

# Auto-transcribe settings
AUTO_TRANSCRIBE = True
AUTO_TRANSCRIBE_WINDOW_S = 12  # fallback window if we can't estimate segment length

# Debounce/re-entry & text dedupe
AUTO_BUSY = False
LAST_TRIGGER_TS = 0.0
MIN_AUTO_GAP_S = 2.5

LAST_TEXT = ""
LAST_TEXT_TS = 0.0
DEDUP_WINDOW_S = 8.0
SIM_THRESHOLD = 0.88


# -----------------------------
# Helpers (audio utils)
# -----------------------------
def _resample_mono_f32(x: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
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
    if x.size == 0:
        return b""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def _safe_label(label: str) -> str:
    label = (label or "").strip().lower()
    label = re.sub(r"[^a-z0-9_-]+", "_", label)
    return label or "snapshot"


def _write_wav(path: str, samplerate: int, mono_f32: np.ndarray) -> None:
    clipped = np.clip(mono_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(samplerate))
        wf.writeframes(pcm16.tobytes())


def _take_latest_samples(seconds: float) -> tuple[np.ndarray, int]:
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
# Text normalize + similarity
# -----------------------------
def _norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s.,!?'\-]", "", s)
    return s


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# -----------------------------
# Capture implementation
# -----------------------------
class _Capture:
    def __init__(self):
        self.stream: sd.InputStream | None = None
        self.lock = threading.Lock()
        self.last_rms: float = 0.0
        self.running: bool = False
        self.samplerate: int = 48000
        self.blocksize: int = max(128, int(self.samplerate * 0.01))
        # Temporary; we resize in start_* based on BUFFER_SECONDS & blocksize
        self.buffer: deque[np.ndarray] = deque(maxlen=1500)

    def _resize_buffer(self):
        # blocks per second ≈ samplerate / blocksize
        bps = max(1, int(round(self.samplerate / max(1, self.blocksize))))
        self.buffer = deque(maxlen=BUFFER_SECONDS * bps)

    def _start_stream(self, **kwargs):
        self.stream = sd.InputStream(**kwargs)
        self.stream.start()
        self.running = True

    def start_mic(self, device_id: int | None, samplerate: int):
        if self.running:
            return
        self.samplerate = samplerate
        self.blocksize = max(128, int(samplerate * 0.01))  # ~10ms blocks for snappy VAD
        self._resize_buffer()
        channels = 1

        def callback(indata, frames, time_info, status):
            mono = indata[:, 0] if indata.ndim == 2 else indata
            mono = mono.astype(np.float32, copy=False)
            with self.lock:
                self.last_rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                self.buffer.append(np.array(mono, dtype=np.float32))

        self._start_stream(
            device=device_id if device_id is not None else None,
            channels=channels,
            samplerate=samplerate,
            blocksize=self.blocksize,
            dtype="float32",
            callback=callback,
        )

    def start_playback_loopback(self, device_id: int | None, samplerate: int):
        import platform
        if self.running:
            return
        if platform.system() != "Windows":
            raise RuntimeError("Loopback capture requires Windows (WASAPI).")
        self.samplerate = samplerate
        self.blocksize = max(256, int(samplerate * 0.02))  # ~20ms blocks
        self._resize_buffer()
        channels = 2
        try:
            wasapi = sd.WasapiSettings(loopback=True)
        except Exception as e:
            raise RuntimeError(f"WASAPI loopback not available: {e}")

        def callback(indata, frames, time_info, status):
            mono = np.mean(indata, axis=1, dtype=np.float32)
            with self.lock:
                self.last_rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                self.buffer.append(np.array(mono, dtype=np.float32))

        self._start_stream(
            device=device_id if device_id is not None else None,
            channels=channels,
            samplerate=samplerate,
            blocksize=self.blocksize,
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
                self.last_rms = 0.0  # keep buffer so /dump_wav still works

    def get_level(self) -> float:
        with self.lock:
            return self.last_rms


_CAPTURE = _Capture()


# -----------------------------
# Whisper (lazy loader)
# -----------------------------
_WHISPER_MODEL = None


def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    model_name = os.environ.get("WHISPER_MODEL", "base")
    _WHISPER_MODEL = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def _ensure_tmp_dir() -> Path:
    out_dir = Path(__file__).resolve().parent.parent / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _dump_last_seconds(seconds: int, label: str = "segment") -> Path | None:
    # enforce sensible bounds
    seconds = int(max(1, min(seconds, BUFFER_SECONDS)))
    with _CAPTURE.lock:
        samplerate = _CAPTURE.samplerate
        blocks = list(_CAPTURE.buffer)
    if not blocks:
        return None
    buf = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    take = int(seconds * samplerate)
    if buf.shape[0] > take:
        buf = buf[-take:]
    out_dir = _ensure_tmp_dir()
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"{_safe_label(label)}_{ts}"
    out = out_dir / f"{base}.wav"
    i = 2
    while out.exists():
        out = out_dir / f"{base}_{i}.wav"
        i += 1
    _write_wav(str(out), samplerate, buf)
    return out


def _transcribe_file(path: Path, language: str | None = "en", beam_size: int = 8) -> Dict[str, Any]:
    """
    Stronger decoding config. Returns:
      - segments: [{start, end, text}, ...]
      - confidence: crude 0..1 from avg_logprob
    """
    model = get_whisper_model()
    segments, info = model.transcribe(
        str(path),
        language=language or "en",
        beam_size=int(beam_size),
        temperature=[0.0, 0.2, 0.4],
        best_of=3,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=True,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
    out = []
    logprobs = []
    for seg in segments:
        text = seg.text.strip()
        out.append({"start": float(seg.start), "end": float(seg.end), "text": text})
        lp = getattr(seg, "avg_logprob", None)
        if lp is not None:
            logprobs.append(float(lp))
    conf = None
    if logprobs:
        avg_lp = sum(logprobs) / len(logprobs)
        conf = max(0.0, min(1.0, 1.0 + avg_lp))  # -1→0, 0→1
    return {"ok": True, "language": info.language, "duration": float(info.duration), "segments": out, "confidence": conf}


def _append_transcript_and_db(filename: str, segs: List[Dict[str, Any]], confidence: float | None):
    """
    Append to in-memory (debug) AND insert into SQLite for CURRENT_MEETING_ID,
    with near-duplicate dedupe.
    """
    global NEXT_UTTER_ID, CURRENT_MEETING_ID, LAST_TEXT, LAST_TEXT_TS

    ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw_text = " ".join(s["text"] for s in segs).strip()
    if not raw_text:
        return

    norm_text = _norm_text(raw_text)
    now_s = time.monotonic()

    # dedupe against very-similar recent insert
    if LAST_TEXT:
        sim = _similar(norm_text, LAST_TEXT)
        if sim >= SIM_THRESHOLD and (now_s - LAST_TEXT_TS) <= DEDUP_WINDOW_S:
            print(f"[dedupe] Skipping near-duplicate (sim={sim:.2f})")
            return

    start_ms = int(segs[0]["start"] * 1000) if segs and "start" in segs[0] else None
    end_ms = int(segs[-1]["end"] * 1000) if segs and "end" in segs[-1] else None

    # 1) in-memory (debug)
    with TRANSCRIPTS_LOCK:
        TRANSCRIPTS.append({
            "id": NEXT_UTTER_ID,
            "ts_iso": ts_iso,
            "filename": filename,
            "start_s": segs[0]["start"] if segs else None,
            "end_s": segs[-1]["end"] if segs else None,
            "text": raw_text,
            "confidence": confidence,
        })
        NEXT_UTTER_ID += 1

    # 2) DB persist (only if meeting active)
    try:
        meeting_id = CURRENT_MEETING_ID
        if meeting_id is not None:
            insert_utterance(
                meeting_id=meeting_id,
                text=raw_text,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=confidence,
                filename=filename,
            )
            LAST_TEXT = norm_text
            LAST_TEXT_TS = now_s
    except Exception as e:
        print(f"[warn] DB insert failed for {filename}: {e}")


def _auto_dump_and_transcribe(seconds: int = AUTO_TRANSCRIBE_WINDOW_S, language: str | None = None):
    """Run in a background thread after 'speech ended' (debounced, re-entry safe)."""
    global AUTO_BUSY, LAST_TRIGGER_TS
    if AUTO_BUSY:
        return
    AUTO_BUSY = True
    try:
        now_m = time.monotonic()
        if (now_m - LAST_TRIGGER_TS) < MIN_AUTO_GAP_S:
            return

        # cap by BUFFER_SECONDS and AUTO_MAX_WINDOW_S
        seconds = int(max(1, min(seconds, AUTO_MAX_WINDOW_S, BUFFER_SECONDS)))

        wav_path = _dump_last_seconds(seconds, label="auto")
        if not wav_path:
            return

        res = _transcribe_file(wav_path, language=language)
        if res.get("ok") and res.get("segments"):
            _append_transcript_and_db(wav_path.name, res["segments"], res.get("confidence"))
            print(f">>> Transcribed {wav_path.name}: {len(res['segments'])} segments")
        else:
            print(f">>> Transcription returned no text for {wav_path.name}")
    except Exception as e:
        print(">>> Auto-transcribe error:", e)
    finally:
        LAST_TRIGGER_TS = time.monotonic()
        AUTO_BUSY = False


# -----------------------------
# Data models
# -----------------------------
class NewMeetingRequest(BaseModel):
    title: str = Field(default="Untitled", description="Meeting title shown in the UI/DB")


class UtteranceIn(BaseModel):
    meeting_id: int | None = None
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    confidence: float | None = None
    filename: str | None = None


class NewMeetingBody(BaseModel):
    title: str


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/devices")
def list_devices():
    info = sd.query_devices()
    hostapis = sd.query_hostapis()
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
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
    playback_id = payload.get("playback_id", None)
    mic_id = payload.get("mic_id", None)
    samplerate = int(payload.get("samplerate", 48000))
    if CAPTURE_STATE["running"]:
        return {"ok": True, "message": "already running", "running": True}
    try:
        if mic_id is not None and mic_id != "":
            _CAPTURE.start_mic(int(mic_id), samplerate)
        else:
            _CAPTURE.start_playback_loopback(int(playback_id) if playback_id is not None else None, samplerate)
        CAPTURE_STATE["running"] = True
        VAD_STATE["speech_active"] = False
        VAD_STATE["speech_frames"] = 0
        VAD_STATE["silence_frames"] = 0
        VAD_STATE["last_voice_ts"] = None
        VAD_STATE["segment_start_ts"] = None
        return {
            "ok": True,
            "message": "started",
            "running": True,
            "playback_id": playback_id,
            "mic_id": mic_id,
            "samplerate": samplerate,
        }
    except Exception as e:
        CAPTURE_STATE["running"] = False
        _CAPTURE.stop()
        return {"ok": False, "message": str(e), "running": False}


@app.post("/stop_capture")
def stop_capture():
    _CAPTURE.stop()
    CAPTURE_STATE["running"] = False
    VAD_STATE["speech_active"] = False
    VAD_STATE["speech_frames"] = 0
    VAD_STATE["silence_frames"] = 0
    VAD_STATE["last_voice_ts"] = None
    VAD_STATE["segment_start_ts"] = None
    return {"ok": True, "message": "stopped", "running": False}


@app.get("/capture_status")
def capture_status():
    return {"running": bool(CAPTURE_STATE["running"])}


@app.get("/level")
def level():
    return {"running": bool(CAPTURE_STATE["running"]), "rms": float(_CAPTURE.get_level())}


@app.get("/dump_wav")
def dump_wav(seconds: int = 5, label: str | None = None):
    # allow bigger manual dumps, but never exceed buffer
    max_allowed = min(300, BUFFER_SECONDS)
    if seconds <= 0 or seconds > max_allowed:
        seconds = min(5, max_allowed)
    with _CAPTURE.lock:
        samplerate = _CAPTURE.samplerate
        blocks = list(_CAPTURE.buffer)
    if not blocks:
        return {"ok": False, "message": "no audio in buffer"}
    buf = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    take = int(seconds * samplerate)
    if buf.shape[0] > take:
        buf = buf[-take:]
    out_dir = _ensure_tmp_dir()
    lbl = _safe_label(label) if label else "snapshot"
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"{lbl}_{ts}"
    out_path = out_dir / f"{base}.wav"
    i = 2
    while out_path.exists():
        out_path = out_dir / f"{base}_{i}.wav"
        i += 1
    _write_wav(str(out_path), samplerate, buf)
    return {
        "ok": True,
        "path": str(out_path.resolve()),
        "filename": out_path.name,
        "seconds": int(seconds),
        "samplerate": int(samplerate),
        "samples": int(buf.shape[0]),
    }


@app.get("/transcribe_wav")
def transcribe_wav(path: str, language: str | None = None, beam_size: int = 5):
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / "tmp" / p
    if not p.exists():
        return {"ok": False, "error": f"File not found: {p}"}
    if p.suffix.lower() != ".wav":
        return {"ok": False, "error": "Only .wav supported"}
    res = _transcribe_file(p, language=language, beam_size=beam_size)
    if res.get("ok") and res.get("segments"):
        _append_transcript_and_db(p.name, res["segments"], res.get("confidence"))
    return res


@app.get("/vad_check")
def vad_check(ng: float = -40.0, vad: int = 3):
    """
    Poll VAD and emit start/end events.
    - Start: short frame-count threshold (snappy).
    - End:    time-based hangover (robust against brief pauses).
    When ending, we estimate the whole segment duration and transcribe that window.
    """
    now = time.monotonic()
    recent, sr = _take_latest_samples(0.1)
    running = bool(CAPTURE_STATE["running"])
    if recent.size == 0:
        return {
            "ok": True,
            "speech": False,
            "rms": 0.0,
            "db": -120.0,
            "running": running,
            "speech_frames": 0,
            "silence_frames": 0,
            "ms_since_voice": 0,
            "events": {"started": False, "ended": False},
        }

    recent_16k = _resample_mono_f32(recent, sr, VAD_SAMPLE_RATE)
    frame_len = int(VAD_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))
    if recent_16k.shape[0] < frame_len:
        rms_short = float(np.sqrt(np.mean(np.square(recent_16k)))) if recent_16k.size else 0.0
        db_short = 20.0 * float(np.log10(max(rms_short, 1e-9)))
        return {
            "ok": True,
            "speech": False,
            "rms": rms_short,
            "db": db_short,
            "running": running,
            "speech_frames": int(VAD_STATE["speech_frames"]),
            "silence_frames": int(VAD_STATE["silence_frames"]),
            "ms_since_voice": int((now - (VAD_STATE.get("last_voice_ts") or now)) * 1000.0),
            "events": {"started": False, "ended": False},
        }

    frame = recent_16k[-frame_len:]
    rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
    db_val = 20.0 * float(np.log10(max(rms, 1e-9)))

    if db_val < ng:
        is_speech_frame = False
    else:
        vad_level = int(max(0, min(3, vad)))
        vad_inst = webrtcvad.Vad(vad_level)
        pcm_bytes = _f32_to_pcm16_bytes(frame)
        is_speech_frame = bool(pcm_bytes) and vad_inst.is_speech(pcm_bytes, VAD_SAMPLE_RATE)

    if is_speech_frame:
        VAD_STATE["last_voice_ts"] = now

    # frame-based counters (for START)
    if is_speech_frame:
        VAD_STATE["speech_frames"] += 1
        VAD_STATE["silence_frames"] = 0
    else:
        VAD_STATE["silence_frames"] += 1
        VAD_STATE["speech_frames"] = 0

    just_started = False
    just_ended = False

    # START: fast
    if not VAD_STATE["speech_active"] and VAD_STATE["speech_frames"] >= START_THRESH_FRAMES:
        VAD_STATE["speech_active"] = True
        VAD_STATE["segment_start_ts"] = now
        just_started = True
        VAD_STATE["last_voice_ts"] = now

    # END: hangover (time since last voiced frame)
    if VAD_STATE["speech_active"]:
        last_voice = VAD_STATE.get("last_voice_ts", now)
        ms_since_voice = (now - last_voice) * 1000.0
        if ms_since_voice >= HANGOVER_MS:
            VAD_STATE["speech_active"] = False
            just_ended = True
            # Estimate the full segment length since it started
            seg_start = VAD_STATE.get("segment_start_ts") or now
            seg_len = int(max(1, min(AUTO_MAX_WINDOW_S, BUFFER_SECONDS, now - seg_start)))
            if AUTO_TRANSCRIBE:
                # Debounced async transcribe of the *whole* segment
                if (now - LAST_TRIGGER_TS) >= MIN_AUTO_GAP_S and not AUTO_BUSY:
                    threading.Thread(
                        target=_auto_dump_and_transcribe,
                        args=(seg_len, None),
                        daemon=True,
                    ).start()

    return {
        "ok": True,
        "running": running,
        "rms": rms,
        "db": db_val,
        "speech": bool(VAD_STATE["speech_active"]),
        "raw_is_speech_frame": bool(is_speech_frame),
        "speech_frames": int(VAD_STATE["speech_frames"]),
        "silence_frames": int(VAD_STATE["silence_frames"]),
        "ms_since_voice": int((now - (VAD_STATE.get("last_voice_ts") or now)) * 1000.0),
        "events": {"started": just_started, "ended": just_ended},
    }


# -----------------------------
# Legacy debug transcript APIs
# -----------------------------
@app.get("/transcripts")
def get_transcripts(since_id: int = 0):
    """Return all in-memory transcripts with id > since_id."""
    with TRANSCRIPTS_LOCK:
        rows = [t for t in TRANSCRIPTS if t["id"] > since_id]
    return {"ok": True, "items": rows, "next_since_id": (rows[-1]["id"] if rows else since_id)}


@app.post("/transcripts/clear")
def clear_transcripts():
    global NEXT_UTTER_ID
    with TRANSCRIPTS_LOCK:
        TRANSCRIPTS.clear()
        NEXT_UTTER_ID = 1
    return {"ok": True}


# -----------------------------
# DB diagnostics
# -----------------------------
@app.get("/db_status")
def db_status():
    """
    Debug endpoint to verify database health.
    """
    from .db import get_connection, DB_PATH
    ok_path = os.path.exists(DB_PATH)
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [r[0] for r in cur.fetchall()]
        return {"ok": True, "db_path": DB_PATH, "exists": ok_path, "tables": tables}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -----------------------------
# Meeting APIs
# -----------------------------
class NewMeetingBody(BaseModel):
    title: str


@app.post("/meeting/new")
def api_meeting_new(body: NewMeetingBody):
    """
    Create a meeting row and return its ID in the field 'meeting_id'
    (the UI expects this exact key).
    """
    mid = new_meeting(body.title)
    return {"ok": True, "meeting_id": mid, "title": body.title}


@app.delete("/meeting/{meeting_id}")
def api_meeting_delete(
    meeting_id: int,
    cascade: bool = Query(
        default=True,
        description="If true, also delete all utterances belonging to this meeting."
    ),
):
    """Delete a meeting by ID."""
    try:
        deleted_meeting, deleted_utt = delete_meeting(meeting_id, cascade=cascade)
    except ValueError as e:
        # cascade=False and there are dependents
        raise HTTPException(status_code=400, detail=str(e))

    if deleted_meeting == 0:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")

    return {
        "ok": True,
        "meeting_id": meeting_id,
        "deleted_meeting": deleted_meeting,
        "deleted_utterances": deleted_utt,
        "cascade": cascade,
    }


@app.post("/meeting/start/{meeting_id}")
def api_meeting_start(meeting_id: int):
    """
    Mark a meeting as 'active'. Subsequent utterance inserts can omit meeting_id.
    """
    # Validate the meeting exists
    try:
        if hasattr(db, "meeting_exists"):
            if not db.meeting_exists(meeting_id):
                raise HTTPException(status_code=404, detail="Meeting not found")
        else:
            # Minimal fallback validation
            with db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1 FROM meetings WHERE id = ?", (meeting_id,))
                if cur.fetchone() is None:
                    raise HTTPException(status_code=404, detail="Meeting not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {e}")

    global CURRENT_MEETING_ID
    CURRENT_MEETING_ID = meeting_id
    return {"ok": True, "id": meeting_id}


@app.post("/meeting/stop")
def api_meeting_stop():
    """
    Clear the active meeting; new utterances will need an explicit meeting_id.
    """
    global CURRENT_MEETING_ID
    CURRENT_MEETING_ID = None
    return {"ok": True}


@app.get("/meeting/active")
def api_meeting_active():
    """
    Return the current active meeting id (or null).
    """
    return {"id": CURRENT_MEETING_ID}


# -----------------------------
# Utterance APIs (for UI)
# -----------------------------
@app.post("/utterance")
def api_insert_utterance(u: UtteranceIn):
    """
    Insert a single utterance row.
    - If meeting_id is omitted, use CURRENT_MEETING_ID.
    """
    meeting_id = u.meeting_id if u.meeting_id is not None else CURRENT_MEETING_ID
    if meeting_id is None:
        raise HTTPException(status_code=400, detail="No meeting_id provided and no active meeting set")

    try:
        new_id = insert_utterance(
            meeting_id=meeting_id,
            text=u.text,
            start_ms=u.start_ms,
            end_ms=u.end_ms,
            confidence=u.confidence,
            filename=u.filename,
        )
        return {"ok": True, "id": new_id, "meeting_id": meeting_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert utterance: {e}")


@app.get("/utterances")
def api_list_utterances(
    meeting_id: int = Query(..., description="Which meeting to fetch"),
    limit: int = Query(200, ge=1, le=500, description="Max rows to return"),
    since_id: int = Query(0, ge=0, description="Only rows with id > since_id"),
):
    """
    List utterances for a meeting with optional delta fetching.

    - If since_id == 0: first page / full snapshot (up to `limit`)
    - Else: only rows with id > since_id (delta)
    Response includes `next_since_id` so clients can resume efficiently.
    """
    rows = db.list_utterances_for_meeting_since(meeting_id, since_id=since_id, limit=limit)

    # Compute next_since_id: the max id we returned (or echo input if none)
    next_since_id = since_id
    for r in rows:
        if r["id"] > next_since_id:
            next_since_id = r["id"]

    return {
        "items": rows,               # ASC by id
        "count": len(rows),
        "next_since_id": next_since_id,
    }