from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
from fastapi import HTTPException, UploadFile
from datetime import datetime, timezone

from ..state import State
from .audio_vad import _resample_mono_f32
import os
from ..db import insert_utterance
import numpy as np


def _ensure_abs_tmp(state: State, p: Path) -> Path:
    return p if p.is_absolute() else (state.tmp_dir / p)


def _save_upload_to_tmp(state: State, upload: UploadFile) -> Path:
    state.tmp_dir.mkdir(parents=True, exist_ok=True)
    name = upload.filename or f"upload.wav"
    out_path = state.tmp_dir / name
    out_path.write_bytes(upload.file.read())
    return out_path


def _get_or_load_model(state: State):
    if state.whisper_model is not None:
        return state.whisper_model  # type: ignore[return-value]
    # Minimal config; can be extended to use Settings later
    from faster_whisper import WhisperModel  # lazy import to avoid test env dependency
    # Harden HF Hub/Faster-Whisper progress handling to avoid tqdm lock issues
    try:
        from contextlib import nullcontext as _nullctx
        import faster_whisper.utils as _fw_utils  # type: ignore
        import faster_whisper.transcribe as _fw_trans  # type: ignore

        class _NoTqdm:
            def __init__(self, iterable=None, **kwargs):
                self.iterable = iterable
            def __iter__(self):
                if self.iterable is None:
                    return iter(())
                for x in self.iterable:
                    yield x
            def update(self, *args, **kwargs):
                return None
            def close(self):
                return None
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        _fw_utils.hf_tqdm = _NoTqdm  # type: ignore[attr-defined]

        # Also disable tqdm usage inside faster_whisper.transcribe by replacing its tqdm symbol
        try:
            _fw_trans.tqdm = _NoTqdm  # type: ignore[attr-defined]
        except Exception:
            pass
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    except Exception:
        pass
    model_dir = os.environ.get("WORKER_WHISPER_MODEL_DIR")
    model_size = os.environ.get("WORKER_WHISPER_MODEL", "base")

    # Prefer on-disk model if valid; otherwise fall back to model name (downloadable)
    target: str = model_size
    if model_dir:
        p = Path(model_dir)
        try:
            has_files = p.is_dir() and (p / "model.bin").exists() and (p / "config.json").exists()
        except Exception:
            has_files = False
        if has_files:
            target = model_dir
        else:
            try:
                import logging
                logging.getLogger("app").warning(
                    f"Ignoring WORKER_WHISPER_MODEL_DIR={model_dir} (missing model.bin/config.json); falling back to '{model_size}'"
                )
            except Exception:
                pass

    state.whisper_model = WhisperModel(target, device="cpu", compute_type="int8")
    return state.whisper_model


def _transcribe_file(state: State, path: Path, language: Optional[str], beam_size: int) -> Dict[str, Any]:
    try:
        model = _get_or_load_model(state)
    except Exception as e:
        try:
            import logging
            logging.getLogger("app").exception("whisper init failed")
        except Exception:
            pass
        return {"ok": False, "error": f"whisper init failed: {e}"}

    # Load audio via soundfile to avoid external ffmpeg dependency
    try:
        import soundfile as sf  # lazy import to avoid CI system lib issues
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y.mean(axis=1).astype(np.float32, copy=False)
        elif not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=np.float32)
        if y.size == 0:
            return {"ok": False, "error": "empty audio"}
        if sr != 16000:
            y = _resample_mono_f32(y, sr, 16000)
        audio_input = y
    except Exception as e:
        try:
            import logging
            logging.getLogger("app").exception("audio read failed")
        except Exception:
            pass
        return {"ok": False, "error": f"audio read failed: {e}"}

    segments, info = model.transcribe(
        audio_input,
        language=language or "en",
        beam_size=int(beam_size),
        temperature=[0.0, 0.2, 0.4],
        best_of=3,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=True,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
    out = []
    logprobs: List[float] = []
    for seg in segments:
        text = seg.text.strip()
        out.append({"start": float(seg.start), "end": float(seg.end), "text": text})
        lp = getattr(seg, "avg_logprob", None)
        if lp is not None:
            logprobs.append(float(lp))
    conf = None
    if logprobs:
        avg_lp = sum(logprobs) / len(logprobs)
        conf = max(0.0, min(1.0, 1.0 + avg_lp))
    return {"ok": True, "language": info.language, "duration": float(info.duration), "segments": out, "confidence": conf}


def _append_transcript_and_db(state: State, filename: str, segs: List[Dict[str, Any]], confidence: Optional[float]) -> Optional[int]:
    ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw_text = " ".join(s["text"] for s in segs).strip()
    if not raw_text:
        return

    # In-memory log
    with state.transcripts.lock:
        state.transcripts.items.append({
            "id": state.transcripts.next_id,
            "ts_iso": ts_iso,
            "filename": filename,
            "start_s": segs[0]["start"] if segs else None,
            "end_s": segs[-1]["end"] if segs else None,
            "text": raw_text,
            "confidence": confidence,
        })
        state.transcripts.next_id += 1

    # Persist (raise on failure so callers can report)
    mid = state.current_meeting_id
    if mid is not None:
        start_ms = int(segs[0]["start"] * 1000) if segs and "start" in segs[0] else None
        end_ms = int(segs[-1]["end"] * 1000) if segs and "end" in segs[-1] else None
        utt_id = insert_utterance(
            meeting_id=mid,
            text=raw_text,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=confidence,
            filename=filename,
        )
        try:
            import logging
            logging.getLogger("app").info(f"inserted utterance id={utt_id} meeting={mid} file={filename}")
        except Exception:
            pass
        return utt_id
    return None


def transcribe_wav(state: State, path: str, language: Optional[str] = None, beam_size: int = 5, meeting_id: Optional[int] = None) -> Dict[str, Any]:
    p = Path(path)
    p = _ensure_abs_tmp(state, p)
    if not p.exists():
        return {"ok": False, "error": f"File not found: {p}"}
    if p.suffix.lower() != ".wav":
        return {"ok": False, "error": "Only .wav supported"}
    res = _transcribe_file(state, p, language=language, beam_size=beam_size)
    # Require at least one segment with text to proceed
    if not (res.get("ok") and res.get("segments")):
        # Fallback attempt: disable internal VAD and relax thresholds
        try:
            mdl = _get_or_load_model(state)

            # Re-load audio here (local to this function)
            import soundfile as sf  # lazy import
            y2, sr2 = sf.read(str(p), dtype="float32", always_2d=False)
            if isinstance(y2, np.ndarray) and y2.ndim == 2:
                y2 = y2.mean(axis=1).astype(np.float32, copy=False)
            elif not isinstance(y2, np.ndarray):
                y2 = np.asarray(y2, dtype=np.float32)
            if y2.size == 0:
                return {"ok": False, "error": "empty audio"}
            if sr2 != 16000:
                y2 = _resample_mono_f32(y2, sr2, 16000)

            segments2, info2 = mdl.transcribe(
                y2,
                language=language or "en",
                beam_size=max(1, int(beam_size)),
                temperature=[0.0],
                best_of=1,
                vad_filter=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.3,
            )
            out2 = []
            logprobs2: List[float] = []
            for seg in segments2:
                text = seg.text.strip()
                if text:
                    out2.append({"start": float(seg.start), "end": float(seg.end), "text": text})
                lp = getattr(seg, "avg_logprob", None)
                if lp is not None:
                    logprobs2.append(float(lp))
            conf2 = None
            if logprobs2:
                avg_lp2 = sum(logprobs2) / len(logprobs2)
                conf2 = max(0.0, min(1.0, 1.0 + avg_lp2))
            if out2:
                res = {"ok": True, "language": getattr(info2, "language", None), "duration": float(getattr(info2, "duration", 0.0)), "segments": out2, "confidence": conf2}
                # Mirror success path: append transcript and return
                orig = state.current_meeting_id
                try:
                    if meeting_id is not None:
                        state.current_meeting_id = meeting_id
                    utt_id = _append_transcript_and_db(state, p.name, res["segments"], res.get("confidence"))
                    if utt_id is not None:
                        res = dict(res)
                        res["utterance_id"] = int(utt_id)
                    return res
                except Exception as e:
                    return {"ok": False, "error": f"db insert failed: {e}"}
                finally:
                    state.current_meeting_id = orig
            else:
                # Provide debug signal metrics
                rms = float(np.sqrt(np.mean(np.square(y2))))
                peak = float(np.max(np.abs(y2)))
                dur = float(len(y2) / 16000.0)
                return {"ok": False, "error": f"No text produced (fallback). rms={rms:.6f} peak={peak:.3f} dur_s={dur:.2f}"}
        except Exception as e:
            try:
                import logging
                logging.getLogger("app").exception("transcribe fallback failed")
            except Exception:
                pass
            # Gracefully degrade if some libs try to use a None context manager
            if isinstance(e, TypeError) and "context manager protocol" in str(e):
                return {"ok": False, "error": "No text produced"}
            return {"ok": False, "error": f"transcribe fallback failed: {e}"}
    else:
        # Temporarily set current meeting if provided
        orig = state.current_meeting_id
        try:
            if meeting_id is not None:
                state.current_meeting_id = meeting_id
            utt_id = _append_transcript_and_db(state, p.name, res["segments"], res.get("confidence"))
            if utt_id is not None:
                res = dict(res)
                res["utterance_id"] = int(utt_id)
            return res
        except Exception as e:
            return {"ok": False, "error": f"db insert failed: {e}"}
        finally:
            state.current_meeting_id = orig


def transcribe_upload(
    state: State,
    file: UploadFile,
    meeting_id: Optional[int] = None,
    language: Optional[str] = None,
    beam_size: int = 8,
) -> Dict[str, Any]:
    mid = meeting_id if meeting_id is not None else state.current_meeting_id
    if mid is None:
        raise HTTPException(status_code=400, detail="No meeting_id provided and no active meeting set")

    p = _save_upload_to_tmp(state, file)
    if p.suffix.lower() != ".wav":
        import soundfile as sf  # lazy import
        data, sr = sf.read(str(p))
        wav_path = p.with_suffix(".wav")
        sf.write(str(wav_path), data, sr)
        p = wav_path

    res = _transcribe_file(state, p, language=language, beam_size=beam_size)
    if not (res.get("ok") and res.get("segments")):
        return {"ok": False, "error": "No text produced"}

    _append_transcript_and_db(state, p.name, res["segments"], res.get("confidence"))

    full_text = " ".join(s["text"].strip() for s in res["segments"]).strip()
    return {"ok": True, "meeting_id": mid, "filename": p.name, "segments": res["segments"], "text": full_text}
