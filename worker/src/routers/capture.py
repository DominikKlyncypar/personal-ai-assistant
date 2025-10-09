from __future__ import annotations

from typing import Any, Dict

import platform

from fastapi import APIRouter, Query, Depends

from ..services import capture as svc
from ..state import get_state, State
from ..models.capture import StartCaptureRequest, StartStopResponse, CaptureStatusResponse, LevelResponse, DumpWavResponse

router = APIRouter(tags=["capture"])


@router.post("/start_capture", response_model=StartStopResponse)
def v1_start_capture(payload: StartCaptureRequest, state: State = Depends(get_state)):
    return svc.start_capture(state, payload.dict())


@router.post("/stop_capture", response_model=StartStopResponse)
def v1_stop_capture(state: State = Depends(get_state)) -> StartStopResponse:
    return svc.stop_capture(state)


@router.get("/capture_status", response_model=CaptureStatusResponse)
def v1_capture_status(state: State = Depends(get_state)) -> CaptureStatusResponse:
    return svc.status(state)


@router.get("/level", response_model=LevelResponse)
def v1_level(state: State = Depends(get_state)) -> LevelResponse:
    return svc.level(state)


@router.get("/dump_wav", response_model=DumpWavResponse)
def v1_dump_wav(seconds: int = 5, label: str | None = None, state: State = Depends(get_state)) -> DumpWavResponse:
    return svc.dump_wav(state, seconds=seconds, label=label)


@router.get("/capture_debug")
def v1_capture_debug(state: State = Depends(get_state)):
    return svc.capture_debug(state)


@router.get("/devices")
def list_devices():
    """Enumerate input/output devices using sounddevice.

    This mirrors the legacy endpoint shape for compatibility with existing UI.
    """
    import sounddevice as sd  # lazy import
    info = sd.query_devices()
    hostapis = sd.query_hostapis()
    try:
        defaults_raw = sd.default.device
    except Exception:
        defaults_raw = (None, None)

    def _norm_default(val):
        try:
            if val is None:
                return None
            iv = int(val)
            return iv if iv >= 0 else None
        except (TypeError, ValueError):
            return None

    default_input = None
    default_output = None
    if isinstance(defaults_raw, (list, tuple)):
        if len(defaults_raw) >= 1:
            default_input = _norm_default(defaults_raw[0])
        if len(defaults_raw) >= 2:
            default_output = _norm_default(defaults_raw[1])
    else:
        default_input = _norm_default(defaults_raw)

    platform_name = platform.system()
    inputs = []
    outputs = []
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
            item_out = dict(item)
            item_out["loopback_capable"] = platform_name == "Windows"
            outputs.append(item_out)
    return {
        "inputs": inputs,
        "outputs": outputs,
        "hostapis": [h["name"] for h in hostapis],
        "defaults": {"input": default_input, "output": default_output},
        "os": platform_name,
    }
