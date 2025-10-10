from __future__ import annotations

from typing import Any, Dict
import time
import platform
from pathlib import Path
import numpy as np
import wave
from collections import deque
import logging

from ..state import State


class Capture:
    def __init__(self, state: State):
        self.state = state
        self.stream: Any | None = None
        self.reader_thread: Any | None = None
        self.logger = logging.getLogger("app.capture")

    def _resize_buffer(self):
        s = self.state.capture
        bps = max(1, int(round(s.samplerate / max(1, s.blocksize))))
        s.buffer = deque(maxlen=self.state.buffer_seconds * bps)

    def _start_stream(self, **kwargs):
        import sounddevice as sd  # lazy import to avoid test env dependency
        # On macOS, prefer a blocking stream + reader thread (more stable with CoreAudio)
        if platform.system() == "Darwin":
            self.stream = sd.InputStream(**{k: v for k, v in kwargs.items() if k != "callback"})
            self.stream.start()
            self.state.capture.running = True

            def _reader():
                s = self.state.capture
                frames = max(128, int(s.samplerate * 0.02))
                while self.state.capture.running and self.stream is not None:
                    try:
                        data, _ = self.stream.read(frames)
                        mono = data[:, 0] if data.ndim == 2 else data
                        mono = mono.astype(np.float32, copy=False)
                        with s.lock:
                            s.last_rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                            s.buffer.append(np.array(mono, dtype=np.float32))
                            s.frames_total += 1
                            try:
                                import time as _t
                                s.last_frame_ts = _t.monotonic()
                            except Exception:
                                s.last_frame_ts = None
                    except Exception:
                        # If read fails (device glitch), sleep briefly and retry
                        try:
                            import time as _t
                            _t.sleep(0.02)
                        except Exception:
                            pass

            import threading as _th
            self.reader_thread = _th.Thread(target=_reader, daemon=True)
            self.reader_thread.start()
        else:
            # Other platforms: use callback stream
            self.stream = sd.InputStream(**kwargs)
            self.stream.start()
            self.state.capture.running = True

    def start_mic(self, device_id: int | None, samplerate: int):
        s = self.state.capture
        if s.running:
            return
        # Preflight device to avoid PortAudio crashes on macOS when settings are invalid
        import sounddevice as sd  # lazy import
        dev_in = device_id if (device_id is not None and device_id >= 0) else None
        # Validate the chosen input device actually has input channels
        try:
            info = sd.query_devices(dev_in)
            if int(info.get('max_input_channels', 0)) <= 0:
                raise RuntimeError("Selected device has no input channels")
        except Exception as e:
            raise RuntimeError(f"Invalid input device: {e}")
        try:
            sd.check_input_settings(device=dev_in, channels=1, samplerate=samplerate, dtype='float32')
        except Exception:
            try:
                sr_default = int(info.get('default_samplerate') or 44100)
                sd.check_input_settings(device=dev_in, channels=1, samplerate=sr_default, dtype='float32')
                samplerate = sr_default
            except Exception as e:
                raise RuntimeError(f"Audio device not usable: {e}")

        s.samplerate = samplerate
        s.blocksize = max(128, int(samplerate * 0.01))
        self._resize_buffer()

        def callback(indata, frames, time_info, status):
            if status:
                try:
                    self.logger.warning("mic callback status=%s", status)
                except Exception:
                    pass
            mono = indata[:, 0] if indata.ndim == 2 else indata
            mono = mono.astype(np.float32, copy=False)
            with s.lock:
                s.last_rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                s.buffer.append(np.array(mono, dtype=np.float32))
                s.frames_total += 1
                try:
                    s.last_frame_ts = time.monotonic()
                except Exception:
                    s.last_frame_ts = None

        self._start_stream(
            device=dev_in,
            channels=1,
            samplerate=samplerate,
            blocksize=s.blocksize,
            dtype="float32",
            latency='high',
            callback=callback,
        )

    def start_playback_loopback(self, device_id: int | None, samplerate: int):
        if self.state.capture.running:
            return
        if platform.system() != "Windows":
            raise RuntimeError("Loopback capture requires Windows (WASAPI).")
        s = self.state.capture
        import sounddevice as sd  # lazy import
        if device_id is None or device_id < 0:
            raise RuntimeError("Select a loopback playback device.")

        try:
            info = sd.query_devices(device_id, "output")
            max_channels = int(info.get("max_output_channels") or 2)
            if max_channels <= 0:
                raise RuntimeError("Selected playback device exposes no channels for loopback capture.")
            samplerate = int(info.get("default_samplerate") or samplerate or 48000)
        except Exception as e:
            raise RuntimeError(f"Loopback device not usable: {e}")

        channels = max(1, min(2, max_channels))

        self.logger.info(
            "start_playback_loopback device=%s hostapi=%s name=%s samplerate=%s channels=%s",
            device_id,
            info.get("hostapi"),
            info.get("name"),
            samplerate,
            channels,
        )

        s.samplerate = samplerate
        s.blocksize = max(256, int(samplerate * 0.02))
        self._resize_buffer()

        try:
            wasapi = sd.WasapiSettings(loopback=True)
        except TypeError:
            try:
                wasapi = sd.WasapiSettings()
                setattr(wasapi, "loopback", True)
            except Exception as e:
                raise RuntimeError(f"WASAPI loopback not available: {e}")
        except Exception as e:
            raise RuntimeError(f"WASAPI loopback not available: {e}")

        def callback(indata, frames, time_info, status):
            try:
                if status:
                    self.logger.warning("loopback callback status=%s", status)
                mono = indata.mean(axis=1).astype(np.float32, copy=False) if indata.ndim == 2 else indata.astype(np.float32, copy=False)
                with s.lock:
                    s.last_rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                    s.buffer.append(np.array(mono, dtype=np.float32))
                    s.frames_total += 1
                    s.last_frame_ts = time.monotonic()
            except Exception:
                self.logger.exception("loopback callback error")

        self._start_stream(
            device=(None, device_id),
            channels=channels,
            samplerate=samplerate,
            blocksize=s.blocksize,
            dtype="float32",
            callback=callback,
            extra_settings=wasapi,
        )

    def stop(self):
        s = self.state.capture
        if not s.running:
            return
        try:
            # Signal threads/streams to stop first
            s.running = False
            if self.stream is not None:
                try:
                    self.stream.stop()
                except Exception:
                    pass
                try:
                    self.stream.close()
                except Exception:
                    pass
        finally:
            self.stream = None
            # Join reader thread if present (best-effort)
            try:
                if self.reader_thread is not None:
                    import threading as _th
                    if isinstance(self.reader_thread, _th.Thread):
                        self.reader_thread.join(timeout=0.2)
            except Exception:
                pass
            self.reader_thread = None
            with s.lock:
                s.last_rms = 0.0


def _ensure_tmp_dir(state: State) -> Path:
    state.tmp_dir.mkdir(parents=True, exist_ok=True)
    return state.tmp_dir


def _write_wav(path: str, samplerate: int, mono_f32: np.ndarray) -> None:
    clipped = np.clip(mono_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(samplerate))
        wf.writeframes(pcm16.tobytes())


def start_capture(state: State, payload: Dict[str, Any]) -> Dict[str, Any]:
    playback_id = payload.get("playback_id", None)
    mic_id = payload.get("mic_id", None)
    samplerate = int(payload.get("samplerate", 48000))

    if state.capture.running:
        return {"ok": True, "message": "already running", "running": True}

    cap = Capture(state)
    try:
        # Normalize blank strings → None
        if isinstance(mic_id, str) and mic_id.strip() == "":
            mic_id = None
        if isinstance(playback_id, str) and playback_id.strip() == "":
            playback_id = None

        # On macOS/Linux, require a real input device; loopback is Windows-only
        started_capture = False
        if platform.system() != "Windows":
            if mic_id is None:
                raise RuntimeError("Select a microphone input. Playback loopback is Windows-only.")
            cap.start_mic(int(mic_id), samplerate)
            started_capture = True
        else:
            # Windows: prefer loopback playback when available; fall back to mic
            if playback_id is not None:
                loop_err: Exception | None = None
                try:
                    import sounddevice as sd  # lazy import

                    dev_idx = int(playback_id)
                    dev_info = sd.query_devices(dev_idx)
                    hostapis = sd.query_hostapis()
                    api_idx = dev_info.get("hostapi", -1) if isinstance(dev_info, dict) else -1
                    api_name = ""
                    if 0 <= api_idx < len(hostapis):
                        try:
                            api_name = str(hostapis[api_idx].get("name") or "")
                        except Exception:
                            api_name = ""
                    name = ""
                    if isinstance(dev_info, dict):
                        try:
                            name = str(dev_info.get("name") or "")
                        except Exception:
                            name = ""
                    cap.logger.info(
                        "loopback candidate id=%s name=%r api=%r",
                        dev_idx,
                        name,
                        api_name,
                    )
                except Exception:
                    pass

                try:
                    cap.start_playback_loopback(int(playback_id), samplerate)
                    started_capture = True
                except Exception as exc:  # Capture the loopback failure
                    loop_err = exc

                if loop_err is not None:
                    if mic_id is not None:
                        cap.logger.warning(
                            "loopback start failed (%s), falling back to mic_id=%s",
                            loop_err,
                            mic_id,
                        )
                        cap.start_mic(int(mic_id), samplerate)
                        started_capture = True
                    else:
                        raise RuntimeError(str(loop_err))
            if not started_capture:
                if mic_id is not None:
                    cap.start_mic(int(mic_id), samplerate)
                    started_capture = True
                else:
                    raise RuntimeError("Select a loopback playback device or microphone before capturing.")
        with state.capture.lock:
            state.capture.running = True
            state.capture.last_error = None
            state.capture.last_error_ts = None
        return {"ok": True, "message": "started", "running": True}
    except Exception as e:  # noqa: BLE001
        with state.capture.lock:
            state.capture.last_error = str(e)
            state.capture.last_error_ts = time.monotonic()
            state.capture.running = False
        try:
            cap.stop()
        except Exception:
            pass
        return {"ok": False, "message": str(e), "running": False}


def stop_capture(state: State) -> Dict[str, Any]:
    cap = Capture(state)
    try:
        cap.stop()
    finally:
        with state.capture.lock:
            state.capture.running = False
    state.vad.speech_active = False
    state.vad.speech_frames = 0
    state.vad.silence_frames = 0
    state.vad.last_voice_ts = None
    state.vad.segment_start_ts = None
    return {"ok": True, "message": "stopped", "running": False}


def status(state: State) -> Dict[str, Any]:
    return {"running": bool(state.capture.running)}


def level(state: State) -> Dict[str, Any]:
    with state.capture.lock:
        rms = float(state.capture.last_rms)
        buf_len = len(state.capture.buffer)
    return {"running": bool(state.capture.running), "rms": rms, "buffer_len": buf_len}


def capture_debug(state: State) -> Dict[str, Any]:
    with state.capture.lock:
        sr = state.capture.samplerate
        bs = state.capture.blocksize
        buf_len = len(state.capture.buffer)
        frames_total = int(state.capture.frames_total)
        last_ts = state.capture.last_frame_ts
        last_err = state.capture.last_error
        last_err_ts = state.capture.last_error_ts
    age_ms = None
    if last_ts is not None:
        try:
            age_ms = int((time.monotonic() - last_ts) * 1000)
        except Exception:
            age_ms = None
    err_age_ms = None
    if last_err_ts is not None:
        try:
            err_age_ms = int((time.monotonic() - last_err_ts) * 1000)
        except Exception:
            err_age_ms = None
    return {
        "running": bool(state.capture.running),
        "samplerate": int(sr),
        "blocksize": int(bs),
        "buffer_blocks": int(buf_len),
        "frames_total": frames_total,
        "ms_since_last_frame": age_ms,
        "last_error": last_err,
        "last_error_age_ms": err_age_ms,
    }


def dump_wav(state: State, seconds: int = 5, label: str | None = None) -> Dict[str, Any]:
    max_allowed = min(300, state.buffer_seconds)
    if seconds <= 0 or seconds > max_allowed:
        seconds = min(5, max_allowed)
    with state.capture.lock:
        samplerate = state.capture.samplerate
        blocks = list(state.capture.buffer)
    if not blocks:
        return {"ok": False, "message": "no audio in buffer"}
    buf = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    take = int(seconds * samplerate)
    if buf.shape[0] > take:
        buf = buf[-take:]
    out_dir = _ensure_tmp_dir(state)
    lbl_input = (label or "snapshot").lower()
    lbl = "".join(c if c.isalnum() or c in "-_" else "_" for c in lbl_input) or "snapshot"
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"{lbl}_{ts}"
    out_path = out_dir / f"{base}.wav"
    i = 2
    while out_path.exists():
        out_path = out_dir / f"{base}_{i}.wav"
        i += 1
    _write_wav(str(out_path), samplerate, buf)
    return {
        "ok": True,
        "path": str(out_path.resolve()),
        "filename": out_path.name,
        "seconds": int(seconds),
        "samplerate": int(samplerate),
        "samples": int(buf.shape[0]),
    }
