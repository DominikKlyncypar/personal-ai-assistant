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

## Windows Build (.exe) & Auto-Update

- The Electron project now uses `electron-builder` and `electron-updater`. To create a local Windows installer run:
  1. Install dependencies: `cd app && npm ci`
  2. Build a Windows virtualenv for the worker (run on Windows):  
     `cd worker && python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt`
  3. Package: `cd app && npm run dist:win`
- GitHub Actions workflow `.github/workflows/release-windows.yml` builds the Windows installer and publishes a release on every push to `main` (and on manual dispatch). Each run bumps the patch version using the workflow run number and uploads installers to GitHub Releases so that shipped builds auto-update.
- To enable publishing from your local machine set `GH_TOKEN` with a GitHub PAT and run `npm run dist:publish`.

## Releases

- Latest: [1.1.20](CHANGELOG.md#1120---2025-10-12) — capitalized/deduped meeting summaries, meeting name status badge, and resilient auto-transcribe fail-stop overlap.
