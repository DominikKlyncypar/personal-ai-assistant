from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os

from .routers.meetings import router as meetings_router
from .routers.capture import router as capture_router
from .routers.transcribe import router as transcribe_router
from .routers.vad import router as vad_router
from .routers.auto import router as auto_router
from .config import load_settings
from .logging import setup_logging, install_app_logging
from .errors import install_error_handlers
from .state import State
from .db import initialize_db


def _load_env_file(env_path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines into os.environ if not set.
    - Ignores comments and blank lines
    - Strips surrounding quotes
    - Supports optional 'export ' prefix
    """
    try:
        if not env_path.exists():
            return
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)
    except Exception:
        # Best-effort only; ignore parse errors
        pass


def create_app() -> FastAPI:
    # Work around tqdm/huggingface progress-bar/thread_map conflicts in some envs
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_MAX_WORKERS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    # Best-effort monkeypatch: make tqdm.contrib.concurrent.ensure_lock a no-op
    try:
        from contextlib import contextmanager
        import tqdm.contrib.concurrent as _tcc  # type: ignore
        import tqdm as _tqdm
        from tqdm import std as _tqdm_std  # type: ignore
        from contextlib import nullcontext as _nullctx

        @contextmanager
        def _safe_ensure_lock(*_args, **_kwargs):
            # Provide a valid no-op context manager to replace tqdm's ensure_lock
            from contextlib import nullcontext as _nullctx
            with _nullctx():
                yield

        _tcc.ensure_lock = _safe_ensure_lock  # type: ignore[attr-defined]

        # Replace thread_map with a simple non-tqdm, single-threaded map to avoid any tqdm usage
        try:
            def _no_tqdm_thread_map(fn, *iterables, **kwargs):  # type: ignore[no-redef]
                if not iterables:
                    return []
                if len(iterables) == 1:
                    it = iterables[0]
                    return [fn(x) for x in it]
                else:
                    return [fn(*args) for args in zip(*iterables)]

            _tcc.thread_map = _no_tqdm_thread_map  # type: ignore[assignment]
        except Exception:
            pass
        # Make tqdm locks/context managers no-ops across std/auto imports
        for mod in (_tqdm, _tqdm_std):
            try:
                mod.tqdm.get_lock = staticmethod(lambda: _nullctx())  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                mod.tqdm.external_write_mode = staticmethod(lambda *a, **k: _nullctx())  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            from tqdm import auto as _tqdm_auto  # type: ignore
            _tqdm_auto.tqdm.get_lock = staticmethod(lambda: _nullctx())  # type: ignore[attr-defined]
            _tqdm_auto.tqdm.external_write_mode = staticmethod(lambda *a, **k: _nullctx())  # type: ignore[attr-defined]
        except Exception:
            pass
        # Ensure the disabled tqdm instance uses a context manager lock
        _disabled = getattr(_tqdm_std, "disabled_tqdm", None)
        if _disabled is not None:
            try:
                setattr(_disabled, "_lock", _nullctx())
                setattr(_disabled, "disable", True)
                setattr(_disabled, "get_lock", staticmethod(lambda: _nullctx()))
            except Exception:
                pass

        # Absolute fallback: ensure any tqdm instance has a usable lock
        try:
            _orig_init = _tqdm_std.tqdm.__init__

            def _safe_tqdm_init(self, *a, **k):  # type: ignore[no-redef]
                _orig_init(self, *a, **k)
                if getattr(self, "_lock", None) is None:
                    self._lock = _nullctx()  # ensure it supports "with"

            _tqdm_std.tqdm.__init__ = _safe_tqdm_init  # type: ignore[assignment]
        except Exception:
            pass
    except Exception as e:
        try:
            import logging
            logging.getLogger("app").warning(f"tqdm monkeypatch skipped: {e}")
        except Exception:
            pass

    # Load environment from optional .env files (repo root and worker dir)
    try:
        src_dir = Path(__file__).resolve().parent
        worker_dir = src_dir.parent
        repo_root = worker_dir.parent
        _load_env_file(repo_root / ".env")
        _load_env_file(worker_dir / ".env")
    except Exception:
        pass

    settings = load_settings()
    setup_logging()

    app = FastAPI(title="Personal AI Assistant Worker", version="1.1.6")

    # Attach config/state
    app.state.settings = settings
    tmp_dir = (Path(__file__).resolve().parent.parent / settings.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    app.state.state = State(tmp_dir=tmp_dir)

    # Ensure database schema exists before handling requests
    try:
        initialize_db()
    except Exception as e:
        try:
            import logging
            logging.getLogger("app").warning(f"initialize_db failed: {e}")
        except Exception:
            pass

    # CORS
    allow = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_app_logging(app)
    install_error_handlers(app)

    # Versioned API
    app.include_router(meetings_router, prefix="/v1")
    app.include_router(capture_router, prefix="/v1")
    app.include_router(transcribe_router, prefix="/v1")
    app.include_router(vad_router, prefix="/v1")
    app.include_router(auto_router, prefix="/v1")

    # Basic health endpoint (for Electron pings)
    @app.get("/health")
    def health():  # pragma: no cover - trivial
        return {"status": "ok"}
    return app


# Convenience for `uvicorn src.app:app`
app = create_app()
