from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict

import os
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        data: Dict[str, Any] = {
            "level": record.levelname,
            "ts": int(time.time() * 1000),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def _resolve_level(default: int = logging.INFO) -> int:
    env_level = os.getenv("WORKER_LOG_LEVEL")
    if not env_level:
        return default
    env_level = env_level.strip()
    if env_level.isdigit():
        try:
            return int(env_level)
        except ValueError:
            return default
    lvl = logging.getLevelName(env_level.upper())
    if isinstance(lvl, str):
        return default
    return int(lvl)


def setup_logging(level: int | None = None) -> None:
    resolved_level = level if level is not None else _resolve_level()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(resolved_level)
    logging.getLogger("app").setLevel(resolved_level)
    logging.getLogger("app.capture").setLevel(resolved_level)
    logging.getLogger("app.access").setLevel(resolved_level)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request.state.request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        dur_ms = int((time.perf_counter() - start) * 1000)
        logging.getLogger("app.access").info(
            json.dumps({
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": dur_ms,
            })
        )
        return response


def install_app_logging(app: FastAPI) -> None:
    app.add_middleware(RequestContextMiddleware)
