from __future__ import annotations

import os
from typing import Optional, Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services.notes import _provider_and_model


class SummaryConfigRequest(BaseModel):
    provider: Optional[str] = Field(default=None, description="auto|openai|groq")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    groq_model: Optional[str] = Field(default=None, description="Groq model id")


router = APIRouter(tags=["summary-config"])


@router.post("/summary_config")
def v1_summary_config(payload: SummaryConfigRequest) -> Dict[str, Any]:
    if payload.provider is not None:
        provider = payload.provider.strip().lower()
        if provider:
            os.environ["WORKER_SUMMARY_PROVIDER"] = provider
        else:
            os.environ.pop("WORKER_SUMMARY_PROVIDER", None)
    if payload.groq_api_key is not None:
        key = payload.groq_api_key.strip()
        if key:
            os.environ["GROQ_API_KEY"] = key
        else:
            os.environ.pop("GROQ_API_KEY", None)
    if payload.groq_model is not None:
        model = payload.groq_model.strip()
        if model:
            os.environ["WORKER_GROQ_MODEL"] = model
        else:
            os.environ.pop("WORKER_GROQ_MODEL", None)

    provider, model_hint = _provider_and_model()
    return {
        "ok": True,
        "provider": provider,
        "model": model_hint,
        "groq_key": bool(os.getenv("GROQ_API_KEY")),
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
    }
