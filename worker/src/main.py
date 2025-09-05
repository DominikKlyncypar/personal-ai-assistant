from __future__ import annotations

# Compatibility shim: keep imports working for src.main:app
# The real application lives in src.app
from .app import app  # noqa: F401

