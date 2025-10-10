from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from fastapi import Request


@dataclass
class CaptureState:
    running: bool = False
    samplerate: int = 48000
    blocksize: int = 480
    last_rms: float = 0.0
    buffer: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=1500))
    lock: threading.Lock = field(default_factory=threading.Lock)
    frames_total: int = 0
    last_frame_ts: Optional[float] = None
    last_error: Optional[str] = None
    last_error_ts: Optional[float] = None


@dataclass
class VadState:
    speech_active: bool = False
    speech_frames: int = 0
    silence_frames: int = 0
    last_voice_ts: Optional[float] = None
    segment_start_ts: Optional[float] = None


@dataclass
class TranscriptsState:
    items: List[Dict[str, Any]] = field(default_factory=list)
    next_id: int = 1
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_text_norm: str = ""
    last_text_ts: float = 0.0


@dataclass
class State:
    """Mutable application state shared across services.

    This replaces module-level globals and is attached to FastAPI's app.state.
    """

    tmp_dir: Path
    buffer_seconds: int = 180
    auto_transcribe: bool = False
    auto_window_s: int = 60
    auto_gap_s: float = 2.5
    auto_failstop_s: int = 180
    auto_failstop_overlap_s: float = 2.0
    auto_busy: bool = False
    last_trigger_ts: float = 0.0

    capture: CaptureState = field(default_factory=CaptureState)
    vad: VadState = field(default_factory=VadState)
    transcripts: TranscriptsState = field(default_factory=TranscriptsState)
    current_meeting_id: Optional[int] = None

    # Whisper model cache
    whisper_model: Any | None = None


def get_state(request: Request) -> State:  # FastAPI dependency helper
    return request.app.state.state
