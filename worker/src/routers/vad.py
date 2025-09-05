from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..state import get_state, State
from ..config import Settings
from ..services import audio_vad
from ..models.vad import VadCheckResponse


router = APIRouter(tags=["vad"])


@router.get("/vad_check", response_model=VadCheckResponse)
def v1_vad_check(ng: float = -40.0, vad: int = 3, request: Request = None, state: State = Depends(get_state)) -> VadCheckResponse:
    settings: Settings = request.app.state.settings  # type: ignore[assignment]
    return audio_vad.vad_check(state, settings, ng=ng, vad=vad)
