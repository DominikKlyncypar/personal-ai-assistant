from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings.

    Values may be provided via environment variables. This keeps configuration
    decoupled from code and simplifies packaging.
    """

    # HTTP
    cors_allow_origins: str = Field("*", description="Comma-separated origins")

    # Audio
    samplerate: int = 48000
    vad_samplerate: int = 16000
    vad_frame_ms: int = 20
    vad_hangover_ms: int = 2500

    # Transcription
    whisper_model_size: str = Field("base", description="e.g. tiny|base|small|medium|large-v2")
    whisper_device: str = Field("auto", description="cpu|cuda|auto")

    # Paths
    tmp_dir: str = Field("tmp", description="Working directory for temp files")

    class Config:
        env_prefix = "WORKER_"
        case_sensitive = False


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
