"""FastAPI routers for the worker.

Routers are grouped by domain (meetings, capture, export, etc.).
"""

from fastapi import APIRouter

# Shared top-level router (optional pattern for grouping)
api_router = APIRouter()

