# Personal AI Assistant

Local desktop assistant for capturing meetings, transcribing audio, and generating readable notes.

## Highlights
- Windows system audio capture via WASAPI loopback.
- Mic capture with auto-transcribe on speech end.
- Meeting summaries with Groq or OpenAI (user-provided key).
- Notes export (Markdown/DOCX) and transcript history.

## Tech Stack
- Electron app in `app/` (HTML/CSS/JS)
- Python FastAPI worker in `worker/`

## Run (dev)

Prereqs:
- Node.js 20+
- Python 3.11

Worker (Windows PowerShell):
```
cd worker
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

Worker (macOS/Linux):
```
cd worker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

App:
```
cd app
npm ci
npm start
```

## Summaries (Groq/OpenAI)
Open Advanced settings in the app and set your provider + API key. Keys are stored locally and sent to the local worker only.

## Build Windows Installer
```
cd app
npm run dist:win
```

## Release Builds (GitHub)
The workflow `.github/workflows/release-windows.yml` publishes Windows installers on pushes to `main`.
Set `GH_TOKEN` as a repo secret with `contents: write` permission.

## Changelog
See `CHANGELOG.md`.
