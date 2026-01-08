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
    def start_playback_loopback(self, device_id: int | str | None, samplerate: int):
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

        try:
            hostapis = sd.query_hostapis()
            wasapi_idx = None
            for i, ha in enumerate(hostapis):
                if "wasapi" in str(ha.get("name") or "").lower():
                    wasapi_idx = i
                    break
            if wasapi_idx is not None:
                sd.default.hostapi = wasapi_idx
        except Exception:
            pass


        if device_id is None or (isinstance(device_id, int) and device_id < 0):
            raise RuntimeError("Select a loopback playback device (OUTPUT) before starting.")


        speaker_name = None
        dev_id: int | None = None
        if isinstance(device_id, str):
            if device_id.startswith("sc:"):
                speaker_name = device_id[3:].strip()
            else:
                try:
                    dev_id = int(device_id)
                except Exception:
                    raise RuntimeError(f"Invalid playback device id: {device_id}")
        else:
            dev_id = int(device_id)
        multi_source = bool(self.streams)
        effective_samplerate = samplerate
        if multi_source and s.samplerate:
            effective_samplerate = int(s.samplerate)
        samplerate = effective_samplerate
        success = False
        # Prefer native WASAPI loopback via soundcard when available (Audacity-like)
        try:
            import soundcard as sc  # type: ignore
        except Exception:
            sc = None
        if sc is None and speaker_name:
            raise RuntimeError("soundcard not installed; cannot use sc: playback id")
        if sc is not None:
            try:
                try:
                    import pythoncom  # type: ignore
                    pythoncom.CoInitialize()
                except Exception:
                    pass
                # Resolve speaker by name (sounddevice -> soundcard)
                try:
                    if dev_id is not None:
                        dev_any = sd.query_devices(dev_id)
                        dev_name = str(dev_any.get("name") or "")
                        max_out_sc = int(dev_any.get("max_output_channels") or 0)
                    else:
                        dev_name = speaker_name or ""
                        max_out_sc = 0
                except Exception:
                    dev_name = ""
                    max_out_sc = 0

                base_norm = _norm_name(dev_name)
                def _paren_token(val: str) -> str:
                    if not val:
                        return ""
                    left = val.rfind("(")
                    right = val.rfind(")")
                    if 0 <= left < right:
                        return _norm_name(val[left + 1:right])
                    return ""

                base_tokens = set(base_norm.split())
                base_paren = _paren_token(dev_name)
                speaker = None
                best_score = -1
                best_name = ""
                speakers = sc.all_speakers()
                for sp in speakers:
                    sp_name = getattr(sp, "name", None) or str(sp)
                    sp_norm = _norm_name(str(sp_name))
                    if speaker_name:
                        if _norm_name(speaker_name) != sp_norm:
                            continue
                        speaker = sp
                        best_name = str(sp_name)
                        best_score = 999
                        break
                    sp_tokens = set(sp_norm.split())
                    score = len(base_tokens & sp_tokens)
                    sp_paren = _paren_token(str(sp_name))
                    if base_paren and sp_paren and (base_paren in sp_paren or sp_paren in base_paren):
                        score += 5
                    if score > best_score:
                        best_score = score
                        speaker = sp
                        best_name = str(sp_name)
                if speaker_name and speaker is None:
                    names = [str(getattr(sp, "name", None) or sp) for sp in speakers]
                    raise RuntimeError(f"soundcard speaker not found: {speaker_name}. available={names}")
                if speaker is None or best_score <= 0:
                    speaker = sc.default_speaker()
                    best_name = str(getattr(speaker, "name", None) or speaker)

                mic = None
                mic_name = None
                try:
                    mic = sc.get_microphone(best_name, include_loopback=True)
                    mic_name = str(getattr(mic, "name", None) or mic)
                except Exception:
                    mic = None
                    mic_name = None
                if mic is None:
                    for m in sc.all_microphones(include_loopback=True):
                        mname = str(getattr(m, "name", None) or m)
                        if _norm_name(best_name) == _norm_name(mname):
                            mic = m
                            mic_name = mname
                            break
                if mic is None:
                    raise RuntimeError(f"soundcard loopback mic not found for speaker: {best_name}")

                if not multi_source:
                    s.samplerate = int(effective_samplerate)
                    s.blocksize = max(256, int(s.samplerate * 0.02))
                    self._resize_buffer()
                self._register_source("loopback")
                self.streams["loopback"] = None
                s.running = True
                with s.lock:
                    s.loopback_backend = "soundcard"
                    s.loopback_device = mic_name or best_name

                def _reader_sc():
                    try:
                        import pythoncom  # type: ignore
                        pythoncom.CoInitialize()
                    except Exception:
                        pass
                    frames = max(256, int(s.samplerate * 0.02))
                    ch_candidates = []
                    if max_out_sc > 0:
                        ch_candidates.append(max_out_sc)
                    ch_candidates += [2, 1]
                    last_err = None
                    for ch in ch_candidates:
                        try:
                            with mic.recorder(
                                samplerate=int(s.samplerate),
                                channels=int(ch),
                                blocksize=None,
                            ) as rec:
                                while self.state.capture.running and "loopback" in self.streams:
                                    data = rec.record(frames)
                                    if data is None:
                                        time.sleep(0.01)
                                        continue
                                    mono = data.mean(axis=1) if getattr(data, "ndim", 1) == 2 else data
                                    self._emit_audio("loopback", mono.astype(np.float32, copy=False))
                            return
                        except Exception as e:
                            last_err = e
                            time.sleep(0.05)
                    if last_err is not None:
                        with s.lock:
                            s.last_error = str(last_err)
                            s.last_error_ts = time.monotonic()

                thread = _th.Thread(target=_reader_sc, daemon=True)
                self.reader_threads["loopback"] = thread
                thread.start()
                success = True
                return
            except Exception as e:
                if speaker_name:
                    raise RuntimeError(f"soundcard loopback failed for {speaker_name}: {e}")
                try:
                    self.logger.warning("soundcard loopback failed, falling back to PortAudio: %s", e)
                except Exception:
                    pass


        if dev_id is None:
            raise RuntimeError("soundcard loopback failed and PortAudio loopback requires a numeric playback id.")

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

            # Prefer explicit WASAPI loopback input devices (Audacity-style) when present
            try:
                total = len(sd.query_devices())
                base_norm = _norm_name(name)
                for idx in range(total):
                    try:
                        d = sd.query_devices(idx)
                        if int(d.get("max_input_channels", 0)) <= 0:
                            continue
                        nm = str(d.get("name") or "")
                        nm_low = nm.lower()
                        if "(loopback)" not in nm_low:
                            continue
                        nm_norm = _norm_name(nm)
                        if base_norm in nm_norm or nm_norm in base_norm:
                            self.logger.info("loopback: using loopback input idx=%s name=%s", idx, nm)
                            self.start_mic(idx, samplerate, source="loopback")
                            success = True
                            return
                    except Exception:
                        continue
            except Exception:
                pass


            def _to_mono_f32(arr):
                data = np.asarray(arr)
                if data.ndim == 2:
                    mono = data.mean(axis=1)
                else:
                    mono = data
                if mono.dtype == np.int16:
                    return mono.astype(np.float32) / 32768.0
                return mono.astype(np.float32, copy=False)


            def _loopback_callback(indata, frames, time_info, status):
                try:
                    if status:
                        self.logger.warning("loopback callback status=%s", status)
                    self._emit_audio("loopback", _to_mono_f32(indata))
                except Exception as e:
                    with s.lock:
                        s.last_error = str(e)
                        s.last_error_ts = time.monotonic()


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


                dtype_candidates = ["float32", "int16"]
                for dtype in dtype_candidates:
                    for ch in ch_candidates:
                        for sr in sr_candidates:
                            try:
                                self.logger.info(
                                    "loopback: trying callback dev=%s ch=%s sr=%s dtype=%s",
                                    dev_idx,
                                    ch,
                                    sr,
                                    dtype,
                                )
                                stream = sd.InputStream(
                                    device=dev_idx,
                                    channels=ch,
                                    samplerate=sr,
                                    blocksize=None,
                                    dtype=dtype,
                                    latency="low",
                                    extra_settings=extra_settings,
                                    callback=_loopback_callback,
                                )
                                stream.start()
                                return (True, ch, sr, stream, True)
                            except Exception as e:
                                last_err = e
                                self.logger.warning(
                                    "loopback callback open failed (dev=%s ch=%s sr=%s dtype=%s): %s",
                                    dev_idx,
                                    ch,
                                    sr,
                                    dtype,
                                    e,
                                )
                            try:
                                self.logger.info(
                                    "loopback: trying blocking dev=%s ch=%s sr=%s dtype=%s",
                                    dev_idx,
                                    ch,
                                    sr,
                                    dtype,
                                )
                                stream = sd.InputStream(
                                    device=dev_idx,  # can be None for default
                                    channels=ch,
                                    samplerate=sr,
                                    blocksize=None,
                                    dtype=dtype,
                                    latency="low",
                                    extra_settings=extra_settings,
                                )
                                stream.start()
                                return (True, ch, sr, stream, False)
                            except Exception as e:
                                last_err = e
                                self.logger.warning(
                                    "loopback open failed (dev=%s ch=%s sr=%s dtype=%s): %s",
                                    dev_idx,
                                    ch,
                                    sr,
                                    dtype,
                                    e,
                                )
                return (False, None, None, last_err, False)


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
                except Exception:
                    try:
                        wasapi_lb = sd.WasapiSettings()
                        setattr(wasapi_lb, "loopback", True)
                        setattr(wasapi_lb, "exclusive", False)
                    except Exception as e:
                        return (False, None, None, RuntimeError(f"WASAPI loopback not supported: {e}"), False)
                ch_list = []
                if isinstance(max_out, int) and max_out > 0:
                    ch_list.append(max_out)
                ch_list += [2, 1, 8, 6, 4, 3]
                return _try_open(
                    target_dev,  # can be None for default
                    ch_list=ch_list,
                    sr_list=sr_base,
                    extra_settings=wasapi_lb,
                )


            # Try preferred path first
            tried: List[str] = []
            ok = False
            ch = sr = res = None
            used_cb = False


            if already_loopback_input:
                tried.append("as_input:selected")
                ok, ch, sr, res, used_cb = _open_as_input(dev_id)
                if not ok:
                    tried.append("as_input:selected")
                    ok, ch, sr, res, used_cb = _open_as_input(dev_id)
                if not ok:
                    tried.append("with_loopback_flag:selected")
                    ok, ch, sr, res, used_cb = _open_with_loopback_flag(dev_id)
            else:
                tried.append("with_loopback_flag:selected")
                ok, ch, sr, res, used_cb = _open_with_loopback_flag(dev_id)
                if not ok:
                    tried.append("as_input:selected")
                    ok, ch, sr, res, used_cb = _open_as_input(dev_id)


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
                        ok, ch, sr, res, used_cb = _open_as_input(idx)
                        if ok:
                            break
                    if not ok:
                        for idx in sibling_indices:
                            tried.append(f"with_loopback_flag:sibling:{idx}")
                            ok, ch, sr, res, used_cb = _open_with_loopback_flag(idx)
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
                        ok, ch, sr, res, used_cb = _open_with_loopback_flag(def_out)
                except Exception:
                    pass


            if not ok:
                tried.append("with_loopback_flag:default_output(None)")
                ok, ch, sr, res, used_cb = _open_with_loopback_flag(None)

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
            with s.lock:
                s.loopback_backend = "portaudio"
                s.loopback_device = name

            if used_cb:
                self.reader_threads["loopback"] = None
                success = True
                return


            def _reader():
                frames = max(256, int(s.samplerate * 0.02))
                error_count = 0
                while self.state.capture.running and "loopback" in self.streams:
                    try:
                        data, _ = stream.read(frames)
                        self._emit_audio("loopback", _to_mono_f32(data))
                        error_count = 0
                    except Exception as e:
                        error_count += 1
                        with s.lock:
                            s.last_error = str(e)
                            s.last_error_ts = time.monotonic()
                        if error_count >= 5:
                            self.logger.warning("loopback read failed repeatedly; disabling loopback source")
                            try:
                                stream.stop()
                            except Exception:
                                pass
                            try:
                                stream.close()
                            except Exception:
                                pass
                            self.streams.pop("loopback", None)
                            self.reader_threads.pop("loopback", None)
                            self._unregister_source("loopback")
                            break
                        time.sleep(0.05)


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
        streams = list(self.streams.items())
        try:
            s.running = False
            for _, stream in streams:
                if stream is None:
                    continue
                try:
                    if hasattr(stream, "abort"):
                        stream.abort()
                except Exception:
                    pass
                try:
                    stream.stop()
                except Exception:
                    pass
            for thread in list(self.reader_threads.values()):
                if isinstance(thread, _th.Thread):
                    try:
                        thread.join(timeout=0.5)
                    except Exception:
                        pass
            for _, stream in streams:
                if stream is None:
                    continue
                try:
                    stream.close()
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
                s.loopback_backend = None
                s.loopback_device = None




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
        if isinstance(mic_id, str):
            if mic_id.strip() == "":
                mic_id = None
            else:
                try:
                    mic_id = int(mic_id)
                except Exception:
                    raise RuntimeError(f"Invalid mic_id: {mic_id}")
        if isinstance(playback_id, str):
            if playback_id.strip() == "":
                playback_id = None
            else:
                try:
                    playback_id = int(playback_id)
                except Exception:
                    # Allow non-numeric IDs (e.g. soundcard loopback speaker ids)
                    playback_id = playback_id.strip()


        if platform.system() != "Windows":
            if mic_id is None:
                raise RuntimeError("Select a microphone input. Playback loopback is Windows-only.")
            cap.start_mic(int(mic_id), samplerate)
        else:
            playback_started = False
            mic_started = False
            loop_err: Exception | None = None
            mic_err: Exception | None = None
            partial_error = False

            if playback_id is None and mic_id is None:
                raise RuntimeError("Select a loopback playback device or microphone before capturing.")

            if playback_id is not None:
                try:
                    cap.start_playback_loopback(playback_id, samplerate)
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
                partial_error = True
                with state.capture.lock:
                    state.capture.last_error = f"loopback start failed: {loop_err}"
                    state.capture.last_error_ts = time.monotonic()
                cap.logger.warning("loopback start failed (%s); continuing with microphone only", loop_err)
            if mic_err is not None and mic_id is not None and playback_started:
                partial_error = True
                with state.capture.lock:
                    state.capture.last_error = f"microphone start failed: {mic_err}"
                    state.capture.last_error_ts = time.monotonic()
                cap.logger.warning("microphone start failed (%s); continuing with loopback only", mic_err)



        with state.capture.lock:
            state.capture.running = True
            if not partial_error:
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
        "loopback_backend": state.capture.loopback_backend,
        "loopback_device": state.capture.loopback_device,
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
