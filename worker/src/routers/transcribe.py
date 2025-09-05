from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from ..services import transcriber as svc
from ..state import get_state, State
from ..models.transcribe import TranscribeWavResponse, TranscribeUploadResponse

router = APIRouter(tags=["transcribe"])


@router.get("/transcribe_wav", response_model=TranscribeWavResponse)
def v1_transcribe_wav(
    path: str,
    language: Optional[str] = None,
    beam_size: int = 5,
    meeting_id: Optional[int] = None,
    state: State = Depends(get_state),
) -> TranscribeWavResponse:
    return svc.transcribe_wav(state, path=path, language=language, beam_size=beam_size, meeting_id=meeting_id)


@router.post("/transcribe_upload", response_model=TranscribeUploadResponse)
def v1_transcribe_upload(
    file: UploadFile = File(...),
    meeting_id: int | None = None,
    language: str | None = None,
    beam_size: int = 8,
    state: State = Depends(get_state),
) -> TranscribeUploadResponse:
    return svc.transcribe_upload(state, file=file, meeting_id=meeting_id, language=language, beam_size=beam_size)
