from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeWavResponse(BaseModel):
    ok: bool
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: List[Segment] = []
    confidence: Optional[float] = None
    error: Optional[str] = None


class TranscribeUploadResponse(BaseModel):
    ok: bool
    meeting_id: int
    filename: str
    segments: List[Segment]
    text: str
