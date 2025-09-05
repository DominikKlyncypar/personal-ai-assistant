from __future__ import annotations

from typing import Optional, Union
from pydantic import BaseModel, Field


class StartCaptureRequest(BaseModel):
    mic_id: Optional[Union[int, str]] = Field(
        default=None,
        description="Input device id (number or string). Empty/None means loopback mode.",
    )
    playback_id: Optional[int] = Field(default=None, description="Output device id for loopback (Windows)")
    samplerate: int = Field(default=48000, ge=8000, le=192000)


class StartStopResponse(BaseModel):
    ok: bool
    message: str
    running: bool


class CaptureStatusResponse(BaseModel):
    running: bool


class LevelResponse(BaseModel):
    running: bool
    rms: float


class DumpWavResponse(BaseModel):
    ok: bool
    path: str
    filename: str
    seconds: int
    samplerate: int
    samples: int
