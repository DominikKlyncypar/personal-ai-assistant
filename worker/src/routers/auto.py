from __future__ import annotations

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..state import State, get_state


class AutoConfigRequest(BaseModel):
    auto_transcribe: Optional[bool] = Field(default=None, description="Enable/disable backend auto-transcribe")
    auto_window_s: Optional[int] = Field(default=None, ge=1, le=300, description="Max seconds per auto chunk")
    auto_gap_s: Optional[float] = Field(default=None, ge=0.0, le=30.0, description="Cooldown between auto runs")
    vad_hangover_ms: Optional[int] = Field(default=None, ge=200, le=10000, description="Silence hangover before segment end (ms)")


router = APIRouter(tags=["auto-config"])


@router.post("/auto_config")
def v1_set_auto_config(payload: AutoConfigRequest, state: State = Depends(get_state), request: Request = None) -> Dict[str, Any]:
    if payload.auto_transcribe is not None:
        state.auto_transcribe = bool(payload.auto_transcribe)
    if payload.auto_window_s is not None:
        state.auto_window_s = int(payload.auto_window_s)
    if payload.auto_gap_s is not None:
        state.auto_gap_s = float(payload.auto_gap_s)
    if payload.vad_hangover_ms is not None and request is not None:
        try:
            # Update live settings so VAD uses the new hangover immediately
            request.app.state.settings.vad_hangover_ms = int(payload.vad_hangover_ms)  # type: ignore[attr-defined]
        except Exception:
            pass

    return {
        "ok": True,
        "auto_transcribe": state.auto_transcribe,
        "auto_window_s": int(state.auto_window_s),
        "auto_gap_s": float(state.auto_gap_s),
        "vad_hangover_ms": getattr(request.app.state.settings, "vad_hangover_ms", None) if request is not None else None,
    }
