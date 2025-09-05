from __future__ import annotations

from pydantic import BaseModel


class VadEvents(BaseModel):
    started: bool
    ended: bool


class VadCheckResponse(BaseModel):
    ok: bool
    running: bool
    rms: float
    db: float
    speech: bool
    raw_is_speech_frame: bool
    speech_frames: int
    silence_frames: int
    ms_since_voice: int
    events: VadEvents

