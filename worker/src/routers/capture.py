from __future__ import annotations

from typing import Any, Dict

import platform

from fastapi import APIRouter, Query, Depends

from ..services import capture as svc
from ..state import get_state, State
from ..models.capture import StartCaptureRequest, StartStopResponse, CaptureStatusResponse, LevelResponse, DumpWavResponse, MixConfigRequest, MixConfigResponse

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


@router.post("/mix_config", response_model=MixConfigResponse)
def v1_mix_config(payload: MixConfigRequest, state: State = Depends(get_state)) -> MixConfigResponse:
    return svc.set_mix_config(state, mic_gain=payload.mic_gain, loopback_gain=payload.loopback_gain)


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

    def _coerce_device(val):
        if isinstance(val, int) and 0 <= val < len(info):
            return val
        try:
            iv = int(val)
            if 0 <= iv < len(info):
                return iv
        except (TypeError, ValueError):
            pass
        if isinstance(val, str):
            for idx, dev in enumerate(info):
                if str(dev.get("name")) == val:
                    return idx
        return None

    default_input = None
    default_output = None
    if isinstance(defaults_raw, (list, tuple)):
        if len(defaults_raw) >= 1:
            default_input = _coerce_device(defaults_raw[0])
        if len(defaults_raw) >= 2:
            default_output = _coerce_device(defaults_raw[1])
    else:
        default_input = _coerce_device(defaults_raw)

    try:
        default_hostapi = getattr(sd.default, "hostapi", None)
        if isinstance(default_hostapi, int) and 0 <= default_hostapi < len(hostapis):
            ha = hostapis[default_hostapi]
            di = ha.get("default_input_device")
            do = ha.get("default_output_device")
            if isinstance(di, int) and di >= 0:
                default_input = default_input if default_input is not None else di
            if isinstance(do, int) and do >= 0:
                default_output = default_output if default_output is not None else do
    except Exception:
        pass

    platform_name = platform.system()
    inputs_raw = []
    outputs_raw = []
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
            "hostapi": dev_api,
        }
        if item["max_input_channels"] > 0:
            inputs_raw.append(item)
        if item["max_output_channels"] > 0:
            item_out = dict(item)
            is_wasapi = platform_name == "Windows" and item_out["hostapi"].lower() == "windows wasapi"
            name_lower = item_out["name"].lower()
            is_loopback = platform_name == "Windows" and ("loopback" in name_lower)
            item_out["loopback_capable"] = is_wasapi
            item_out["is_loopback"] = is_loopback
            outputs_raw.append(item_out)
        elif platform_name == "Windows" and item["max_input_channels"] > 0:
            name_lower = item["name"].lower()
            if any(tag in name_lower for tag in ["stereo mix", "what u hear", "wave out mix", "mix (realtek", "soundboard"]):
                item_out = dict(item)
                item_out["max_output_channels"] = item_out["max_input_channels"]
                item_out["loopback_capable"] = True
                item_out["is_loopback"] = False
                outputs_raw.append(item_out)

    loopback_inputs = []
    loopback_speakers = []
    if platform_name == "Windows":
        for d in inputs_raw:
            name_lower = str(d.get("name") or "").lower()
            if "(loopback)" in name_lower:
                loopback_inputs.append(dict(d))
        try:
            import soundcard as sc  # type: ignore
            seen = set()
            for sp in sc.all_speakers():
                sp_name = str(getattr(sp, "name", None) or sp)
                key = sp_name.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                loopback_speakers.append({"id": f"sc:{sp_name}", "name": sp_name})
        except Exception:
            pass

    def _filter_by_preferred(devices):
        if platform_name != "Windows":
            return devices
        wasapi = []
        fallback = []
        for d in devices:
            host = str(d.get("hostapi") or "")
            if "wasapi" in host.lower():
                wasapi.append(d)
            else:
                fallback.append(d)
        return wasapi if wasapi else fallback

    def _dedupe(devices):
        seen = set()
        trimmed = []
        for d in devices:
            key = d["name"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            trimmed.append(d)
        return trimmed

    inputs = _dedupe(_filter_by_preferred(inputs_raw))
    outputs = _dedupe(_filter_by_preferred(outputs_raw))

    if platform_name == "Windows":
        outputs.sort(key=lambda d: (0 if d.get("is_loopback") else 1, d["name"].lower()))
    else:
        outputs.sort(key=lambda d: d["name"].lower())
    inputs.sort(key=lambda d: d["name"].lower())

    if platform_name == "Windows":
        loopback_ids = [d["id"] for d in outputs if d.get("is_loopback")]
        if loopback_ids:
            default_output = loopback_ids[0]

    return {
        "inputs": inputs,
        "outputs": outputs,
        "loopback_inputs": loopback_inputs,
        "loopback_speakers": loopback_speakers,
        "hostapis": [h["name"] for h in hostapis],
        "defaults": {"input": default_input, "output": default_output},
        "os": platform_name,
    }
