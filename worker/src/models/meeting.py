from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field


class MeetingMeta(BaseModel):
    id: int
    title: Optional[str] = None
    created_at: Optional[str] = Field(None, description="ISO timestamp if available")


class Utterance(BaseModel):
    id: int
    meeting_id: int
    ts_iso: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    text: str
    confidence: Optional[float] = None
    filename: Optional[str] = None
    speaker: Optional[str] = Field(None, description="Optional speaker label if diarized")


class ExportJSONResponse(BaseModel):
    meeting: MeetingMeta
    utterances: List[Utterance]

