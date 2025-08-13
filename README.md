# Personal AI Assistant

Desktop app built with:
- Electron + React (UI) in `/app`
- Python FastAPI (AI worker) in `/worker`

## Current Status
Phase 0 complete:
- Node/Electron skeleton
- Python venv + FastAPI health
- Shared configs: .editorconfig, Prettier, ESLint, Black, Ruff

## Run (dev)
- UI:
  cd app && npm start
- Worker:
  cd worker && source .venv/bin/activate && uvicorn src.main:app --reload --port 8000
