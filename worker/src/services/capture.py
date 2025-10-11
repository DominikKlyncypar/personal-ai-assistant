from __future__ import annotations


from typing import Any, Dict, List
import time
import platform
from pathlib import Path
import numpy as np
import wave
from collections import deque
import logging
import threading


from ..state import State


# ---- safe lowercase helper ----
def _safe_lower(x):
    try:
        return str(x or "").lower()
    except Exception:
        return ""




def _norm_name(n: str) -> str:
    """Normalize device names for fuzzy matching."""
    s = _safe_lower(n)
    s = s.replace("(loopback)", " ")
    s = s.replace("[windows wasapi]", " ")
    s = s.replace("[wdm-ks]", " ")
    s = s.replace("[mme]", " ")
    s = s.replace("[directsound]", " ")
    s = " ".join(s.split())
    return s




class Capture:
    def __init__(self, state: State):
        self.state = state
        self.logger = logging.getLogger("app.capture")
        self.streams: Dict[str, Any] = {}
        self.reader_threads: Dict[str, Any] = {}
        self._active_sources: List[str] = []
        self._mix_pending: Dict[str, np.ndarray] = {}
        self._mix_lock = threading.Lock()


    def _resize_buffer(self):
        s = self.state.capture
        bs = s.blocksize if isinstance(s.blocksize, int) and s.blocksize > 0 else 1024
        bps = max(1, int(round(s.samplerate / max(1, bs))))
        s.buffer = deque(maxlen=self.state.buffer_seconds * bps)

    def _register_source(self, source: str):
        with self._mix_lock:
            if source not in self._active_sources:
                self._active_sources.append(source)
            if source not in self._mix_pending:
                self._mix_pending[source] = np.empty(0, dtype=np.float32)

    def _unregister_source(self, source: str):
        with self._mix_lock:
            if source in self._active_sources:
                self._active_sources.remove(source)
            self._mix_pending.pop(source, None)

    def _push_to_buffer(self, mono: np.ndarray):
        mono = np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)
        s = self.state.capture
        with s.lock:
            s.last_rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            s.buffer.append(np.array(mono, dtype=np.float32, copy=False))
            s.frames_total += 1
            s.last_frame_ts = time.monotonic()

    def _emit_audio(self, source: str, mono: np.ndarray):
        if mono is None:
            return
        mono = np.asarray(mono, dtype=np.float32)
        chunks: List[np.ndarray] = []
        with self._mix_lock:
            active = [src for src in self._active_sources if src in self.streams]
            if len(active) <= 1:
                chunks.append(mono)
            else:
                pending = self._mix_pending.setdefault(source, np.empty(0, dtype=np.float32))
                if pending.size:
                    self._mix_pending[source] = np.concatenate((pending, mono))
                else:
                    self._mix_pending[source] = mono
                while True:
                    lengths = [len(self._mix_pending[src]) for src in active]
                    if not lengths or min(lengths) == 0:
                        break
                    take = min(lengths)
                    mix = np.zeros(take, dtype=np.float32)
                    for src in active:
                        seg = self._mix_pending[src][:take]
                        mix[: len(seg)] += seg
                        self._mix_pending[src] = self._mix_pending[src][take:]
                    if len(active) > 1:
                        mix /= float(len(active))
                    chunks.append(mix)
        for chunk in chunks:
            if chunk.size:
                self._push_to_buffer(chunk)


    def _start_stream(self, source: str, **kwargs):
        import sounddevice as sd
        self._register_source(source)
        if platform.system() == "Darwin":
            # macOS: blocking InputStream + reader thread
            stream = sd.InputStream(**{k: v for k, v in kwargs.items() if k != "callback"})
            stream.start()
            self.streams[source] = stream
            self.state.capture.running = True


            def _reader():
                s = self.state.capture
                frames = max(128, int(s.samplerate * 0.02))
                while self.state.capture.running and source in self.streams:
                    try:
                        data, _ = stream.read(frames)
                        mono = data[:, 0] if data.ndim == 2 else data
                        self._emit_audio(source, mono)
                    except Exception:
                        time.sleep(0.02)


            import threading as _th
            reader = _th.Thread(target=_reader, daemon=True)
            self.reader_threads[source] = reader
            reader.start()
        else:
            # non-mac: let caller choose path
            stream = sd.InputStream(**kwargs)
            stream.start()
            self.streams[source] = stream
            self.reader_threads[source] = None
            self.state.capture.running = True


    def start_mic(self, device_id: int | None, samplerate: int, source: str = "mic"):
        s = self.state.capture
        if source in self.streams:
            return
        import sounddevice as sd
        dev_in = device_id if (device_id is not None and device_id >= 0) else None
        try:
            info = sd.query_devices(dev_in)
            if int(info.get('max_input_channels', 0)) <= 0:
                raise RuntimeError("Selected device has no input channels")
        except Exception as e:
            raise RuntimeError(f"Invalid input device: {e}")
        multi_source = bool(self.streams)
        target_samplerate = samplerate
        if multi_source and s.samplerate:
            target_samplerate = int(s.samplerate)
        try:
            sd.check_input_settings(device=dev_in, channels=1, samplerate=target_samplerate, dtype='float32')
        except Exception:
            try:
                sr_default = int(info.get('default_samplerate') or target_samplerate or 44100)
                if multi_source and sr_default != target_samplerate:
                    raise RuntimeError(
                        f"Microphone device cannot run at required samplerate {target_samplerate} Hz to match loopback."
                    )
                sd.check_input_settings(device=dev_in, channels=1, samplerate=sr_default, dtype='float32')
                target_samplerate = sr_default
            except Exception as e:
                raise RuntimeError(f"Audio device not usable: {e}")

        samplerate = target_samplerate
        if not multi_source:
            s.samplerate = samplerate
            s.blocksize = max(128, int(samplerate * 0.01))
            if platform.system() == "Windows":
                s.blocksize = max(256, int(samplerate * 0.02))
            self._resize_buffer()
        self._register_source(source)

        if platform.system() == "Windows":
            # Windows mic: blocking InputStream + reader
            wasapi = None
            try:
                hostapis = sd.query_hostapis()
                hai = int(info.get("hostapi", -1))
                api_name = ""
                if 0 <= hai < len(hostapis):
                    api_name = str(hostapis[hai].get("name") or "")
                if "wasapi" in api_name.lower():
                    try:
                        wasapi = sd.WasapiSettings(exclusive=False)
                    except Exception:
                        wasapi = None
            except Exception:
                wasapi = None


            try:
                stream = sd.InputStream(
                    device=dev_in,
                    channels=1,
                    samplerate=samplerate,
                    blocksize=None,
                    dtype="float32",
                    latency="low",
                    extra_settings=wasapi,
                )
                stream.start()
                self.streams[source] = stream
                s.running = True
            except Exception as e:
                self._unregister_source(source)
                raise RuntimeError(f"Error starting stream: {e}")


            def _reader():
                frames = max(256, int(s.samplerate * 0.02))
                while self.state.capture.running and source in self.streams:
                    try:
                        data, _ = stream.read(frames)
                        mono = data[:, 0] if data.ndim == 2 else data
                        self._emit_audio(source, mono)
                    except Exception as e:
                        with s.lock:
                            s.last_error = str(e)
                            s.last_error_ts = time.monotonic()
                        time.sleep(0.02)


            import threading as _th
            thread = _th.Thread(target=_reader, daemon=True)
            self.reader_threads[source] = thread
            thread.start()
            return


        # Non-Windows mic: callback path
        def callback(indata, frames, time_info, status):
            try:
                if status:
                    self.logger.warning("mic callback status=%s", status)
                mono = indata[:, 0] if indata.ndim == 2 else indata
                self._emit_audio(source, mono)
            except Exception as e:
                with s.lock:
                    s.last_error = str(e)
                    s.last_error_ts = time.monotonic()


        stream_kwargs = dict(
            device=dev_in,
            channels=1,
            samplerate=samplerate,
            blocksize=s.blocksize,
            dtype="float32",
            latency='high',
            callback=callback,
        )
        self._start_stream(source, **stream_kwargs)


    # -------- Windows WASAPI loopback with capability-based detection, sibling search & WASAPI default --------
    def start_playback_loopback(self, device_id: int | None, samplerate: int):
        """
        Windows WASAPI loopback with robust detection:
        1) If device looks like a dedicated loopback INPUT (max_in>0 and max_out==0 or name has "(loopback)"),
           open as a normal INPUT (no WasapiSettings(loopback=True)).
        2) Else, open with WasapiSettings(loopback=True).
        3) If both fail, SEARCH for a sibling "(loopback)" input under the same hostapi whose normalized name
           matches the selected device, and try that.
        4) If still failing, try the WASAPI host's **default output device index** with loopback=True.
        5) Finally, try default output via device=None with loopback=True.
        """
        if "loopback" in self.streams:
            return
        if platform.system() != "Windows":
            raise RuntimeError("Loopback capture requires Windows (WASAPI).")


        s = self.state.capture
        import sounddevice as sd
        import threading as _th


        if device_id is None or int(device_id) < 0:
            raise RuntimeError("Select a loopback playback device (OUTPUT) before starting.")


        dev_id = int(device_id)
        multi_source = bool(self.streams)
        effective_samplerate = samplerate
        if multi_source and s.samplerate:
            effective_samplerate = int(s.samplerate)
        samplerate = effective_samplerate
        self._register_source("loopback")
        success = False


        try:
            # --- probe device views ---
            try:
                di_any = sd.query_devices(dev_id)  # generic
                name = str(di_any.get("name") or "")
                hostapi_idx = int(di_any.get("hostapi", -1))
            except Exception as e:
                raise RuntimeError(f"Selected device not found: {e}")


            # input view
            max_in, in_default_sr = 0, None
            try:
                di_in = sd.query_devices(dev_id, "input")
                max_in = int(di_in.get("max_input_channels") or 0)
                in_default_sr = di_in.get("default_samplerate")
            except Exception:
                pass


            # output view
            max_out, out_default_sr = 0, None
            try:
                di_out = sd.query_devices(dev_id, "output")
                max_out = int(di_out.get("max_output_channels") or 0)
                out_default_sr = di_out.get("default_samplerate")
            except Exception:
                pass


            name_lower = name.lower()
            loopback_name_tokens = ("(loopback)", "stereo mix", "wave out mix", "what u hear", "mix (realtek", "soundboard")
            already_loopback_input = (max_in > 0 and max_out == 0) or any(tok in name_lower for tok in loopback_name_tokens)
            if already_loopback_input:
                self.start_mic(dev_id, samplerate, source="loopback")
                success = True
                return


            def _try_open(dev_idx: int | None, ch_list, sr_list, extra_settings):
                last_err = None
                # channel candidates
                ch_candidates: List[int] = []
                for c in ch_list:
                    try:
                        c = int(c)
                        if c > 0 and c not in ch_candidates:
                            ch_candidates.append(c)
                    except Exception:
                        pass
                if not ch_candidates:
                    ch_candidates = [2, 1]
                # samplerate candidates
                sr_candidates: List[int] = []
                for v in sr_list:
                    try:
                        vi = int(v) if v is not None else None
                        if vi and vi not in sr_candidates:
                            sr_candidates.append(vi)
                    except Exception:
                        pass
                if not sr_candidates:
                    sr_candidates = [48000, 44100]


                self.logger.info(
                    "loopback: opening dev_id=%s name=%s ch_candidates=%s sr_candidates=%s loopback_flag=%s",
                    dev_idx,
                    name if dev_idx == dev_id else f"(alt:{dev_idx})",
                    ch_candidates,
                    sr_candidates,
                    (getattr(extra_settings, "loopback", False) if extra_settings is not None else False),
                )


                for ch in ch_candidates:
                    for sr in sr_candidates:
                        try:
                            self.logger.info("loopback: trying dev=%s ch=%s sr=%s", dev_idx, ch, sr)
                            stream = sd.InputStream(
                                device=dev_idx,  # can be None for default
                                channels=ch,
                                samplerate=sr,
                                blocksize=None,
                                dtype="float32",
                                latency="low",
                                extra_settings=extra_settings,
                            )
                            stream.start()
                            return (True, ch, sr, stream)
                        except Exception as e:
                            last_err = e
                            self.logger.warning("loopback open failed (dev=%s ch=%s sr=%s): %s", dev_idx, ch, sr, e)
                return (False, None, None, last_err)


            # Build default sample-rate base (includes both views + requested/effective)
            sr_base = [in_default_sr, out_default_sr, effective_samplerate, 48000, 44100]


            # Path A: open as normal INPUT (no loopback flag) using input caps
            def _open_as_input(target_dev: int):
                ch_list = [2, 1]
                try:
                    mi = int(sd.query_devices(target_dev, "input").get("max_input_channels", 0))
                    if mi > 0:
                        ch_list.insert(0, mi)
                except Exception:
                    pass
                return _try_open(
                    target_dev,
                    ch_list=ch_list,
                    sr_list=sr_base,
                    extra_settings=None,  # IMPORTANT: no loopback flag here
                )


            # Path B: open with WASAPI loopback flag (for regular output endpoints)
            def _open_with_loopback_flag(target_dev: int | None):
                try:
                    wasapi_lb = sd.WasapiSettings(loopback=True, exclusive=False)
                except TypeError:
                    wasapi_lb = sd.WasapiSettings()
                    setattr(wasapi_lb, "loopback", True)
                    setattr(wasapi_lb, "exclusive", False)
                return _try_open(
                    target_dev,  # can be None for default
                    ch_list=[2, 1, 8, 6, 4, 3],
                    sr_list=sr_base,
                    extra_settings=wasapi_lb,
                )


            # Try preferred path first
            tried: List[str] = []
            ok = False
            ch = sr = res = None


            if already_loopback_input:
                tried.append("as_input:selected")
                ok, ch, sr, res = _open_as_input(dev_id)
                if not ok:
                    tried.append("as_input:selected")
                    ok, ch, sr, res = _open_as_input(dev_id)
                if not ok:
                    tried.append("with_loopback_flag:selected")
                    ok, ch, sr, res = _open_with_loopback_flag(dev_id)
            else:
                tried.append("with_loopback_flag:selected")
                ok, ch, sr, res = _open_with_loopback_flag(dev_id)
                if not ok:
                    tried.append("as_input:selected")
                    ok, ch, sr, res = _open_as_input(dev_id)


            # Sibling search
            sibling_indices: List[int] = []
            if not ok:
                try:
                    all_num = len(sd.query_devices())
                    base_norm = _norm_name(name)
                    for idx in range(all_num):
                        try:
                            d = sd.query_devices(idx)
                            if int(d.get("hostapi", -1)) != hostapi_idx:
                                continue
                            nm = str(d.get("name") or "")
                            di_in = sd.query_devices(idx, "input")
                            mi = int(di_in.get("max_input_channels") or 0)
                            if mi <= 0:
                                continue
                            nm_low = nm.lower()
                            nm_norm = _norm_name(nm)
                            if base_norm in nm_norm or nm_norm in base_norm or "(loopback)" in nm_low:
                                sibling_indices.append(idx)
                        except Exception:
                            continue

                    for idx in sibling_indices:
                        tried.append(f"as_input:sibling:{idx}")
                        ok, ch, sr, res = _open_as_input(idx)
                        if ok:
                            break
                    if not ok:
                        for idx in sibling_indices:
                            tried.append(f"with_loopback_flag:sibling:{idx}")
                            ok, ch, sr, res = _open_with_loopback_flag(idx)
                            if ok:
                                break
                except Exception:
                    pass


            if not ok:
                try:
                    hostapis = sd.query_hostapis()
                    wasapi_idx = None
                    for i, ha in enumerate(hostapis):
                        if "wasapi" in str(ha.get("name") or "").lower():
                            wasapi_idx = i
                            break
                    if wasapi_idx is not None:
                        def_out = hostapis[wasapi_idx].get("default_output_device", None)
                    else:
                        def_out = hostapis[hostapi_idx].get("default_output_device", None) if 0 <= hostapi_idx < len(hostapis) else None
                    if isinstance(def_out, int) and def_out >= 0:
                        tried.append(f"with_loopback_flag:wasapi_default:{def_out}")
                        ok, ch, sr, res = _open_with_loopback_flag(def_out)
                except Exception:
                    pass


            if not ok:
                tried.append("with_loopback_flag:default_output(None)")
                ok, ch, sr, res = _open_with_loopback_flag(None)

            if not ok:
                fallback_inputs: List[int] = []
                base_norm = _norm_name(name)
                loopback_tags = ["stereo mix", "what u hear", "wave out mix", "mix (realtek", "loopback", "soundboard"]
                try:
                    total = len(sd.query_devices())
                except Exception:
                    total = 0
                for idx in range(total):
                    try:
                        d = sd.query_devices(idx)
                        if int(d.get("max_input_channels", 0)) <= 0:
                            continue
                        nm = str(d.get("name") or "")
                        nm_low = nm.lower()
                        nm_norm = _norm_name(nm)
                        if any(tag in nm_low for tag in loopback_tags) or base_norm in nm_norm or nm_norm in base_norm:
                            fallback_inputs.append(idx)
                    except Exception:
                        continue
                last_err: Exception | None = None
                for idx in fallback_inputs:
                    try:
                        self.start_mic(idx, samplerate, source="loopback")
                        success = True
                        return
                    except Exception as inner_exc:
                        last_err = inner_exc
                if last_err is not None:
                    res = last_err


            if not ok:
                raise RuntimeError(f"Error starting loopback stream: {res} (tried: {tried})")


            stream = res
            actual_sr = int(sr)
            if multi_source and s.samplerate and actual_sr != int(s.samplerate):
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Loopback device samplerate {actual_sr} Hz does not match microphone samplerate {s.samplerate} Hz."
                )
            s.samplerate = actual_sr
            if not multi_source:
                s.blocksize = max(256, int(s.samplerate * 0.02))
                self._resize_buffer()
            self.streams["loopback"] = stream
            s.running = True


            def _reader():
                frames = max(256, int(s.samplerate * 0.02))
                while self.state.capture.running and "loopback" in self.streams:
                    try:
                        data, _ = stream.read(frames)
                        mono = data.mean(axis=1) if getattr(data, "ndim", 1) == 2 else data
                        self._emit_audio("loopback", mono)
                    except Exception as e:
                        with s.lock:
                            s.last_error = str(e)
                            s.last_error_ts = time.monotonic()
                        time.sleep(0.02)


            thread = _th.Thread(target=_reader, daemon=True)
            self.reader_threads["loopback"] = thread
            thread.start()
            success = True
        except Exception:
            self._unregister_source("loopback")
            raise
        finally:
            if not success:
                stream_obj = self.streams.pop("loopback", None)
                if stream_obj is not None:
                    try:
                        stream_obj.stop()
                    except Exception:
                        pass
                    try:
                        stream_obj.close()
                    except Exception:
                        pass
                self.reader_threads.pop("loopback", None)
                self._unregister_source("loopback")


    def stop(self):
        s = self.state.capture
        if not s.running:
            return
        import threading as _th
        try:
            s.running = False
            for src, stream in list(self.streams.items()):
                if stream is None:
                    continue
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
        finally:
            try:
                for thread in list(self.reader_threads.values()):
                    if isinstance(thread, _th.Thread):
                        try:
                            thread.join(timeout=0.2)
                        except Exception:
                            pass
            finally:
                self.streams.clear()
                self.reader_threads.clear()
                with self._mix_lock:
                    self._active_sources.clear()
                    self._mix_pending.clear()
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
        if isinstance(mic_id, str) and mic_id.strip() == "":
            mic_id = None
        if isinstance(playback_id, str) and playback_id.strip() == "":
            playback_id = None


        if platform.system() != "Windows":
            if mic_id is None:
                raise RuntimeError("Select a microphone input. Playback loopback is Windows-only.")
            cap.start_mic(int(mic_id), samplerate)
        else:
            playback_started = False
            mic_started = False
            loop_err: Exception | None = None
            mic_err: Exception | None = None

            if playback_id is None and mic_id is None:
                raise RuntimeError("Select a loopback playback device or microphone before capturing.")

            if playback_id is not None:
                try:
                    cap.start_playback_loopback(int(playback_id), samplerate)
                    playback_started = True
                    samplerate = state.capture.samplerate or samplerate
                except Exception as exc:
                    loop_err = exc

            if mic_id is not None:
                target_sr = state.capture.samplerate or samplerate
                try:
                    cap.start_mic(int(mic_id), target_sr)
                    mic_started = True
                except Exception as exc:
                    mic_err = exc

            if not playback_started and not mic_started:
                err = loop_err or mic_err or RuntimeError("Unable to start capture.")
                raise RuntimeError(str(err))

            if loop_err is not None and playback_id is not None and mic_started:
                cap.logger.warning("loopback start failed (%s); continuing with microphone only", loop_err)
            if mic_err is not None and mic_id is not None and playback_started:
                cap.logger.warning("microphone start failed (%s); continuing with loopback only", mic_err)



        with state.capture.lock:
            state.capture.running = True
            state.capture.last_error = None
            state.capture.last_error_ts = None
            state.capture.controller = cap
        return {"ok": True, "message": "started", "running": True}
    except Exception as e:
        with state.capture.lock:
            state.capture.last_error = str(e)
            state.capture.last_error_ts = time.monotonic()
            state.capture.running = False
        try:
            cap.stop()
        except Exception:
            pass
        with state.capture.lock:
            state.capture.controller = None
        return {"ok": False, "message": str(e), "running": False}




def stop_capture(state: State) -> Dict[str, Any]:
    cap = state.capture.controller or Capture(state)
    try:
        cap.stop()
    finally:
        with state.capture.lock:
            state.capture.running = False
            state.capture.controller = None
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
        "blocksize": int(bs) if isinstance(bs, int) and bs > 0 else 0,
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
    lbl_input = label or "snapshot"
    lbl = "".join(c if c.isalnum() or c in "-_" else "_" for c in lbl_input.lower()) or "snapshot"
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
