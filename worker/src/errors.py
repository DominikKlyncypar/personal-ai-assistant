from __future__ import annotations

from typing import Any, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _handle_http_exception(request: Request, exc: HTTPException):  # type: ignore[unused-variable]
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=str(exc.detail)).dict(),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception):  # type: ignore[unused-variable]
        return JSONResponse(status_code=500, content=ErrorResponse(error="internal error").dict())

