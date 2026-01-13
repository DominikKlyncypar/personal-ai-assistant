# Changelog

All notable changes to this project will be documented in this file.

## [1.1.26] - 2026-01-09

### Added
- Windows WASAPI loopback capture via soundcard backend and explicit loopback device list.
- Summary provider settings UI with local Groq key storage and /v1/summary_config.
- Mic/system sensitivity sliders with live mix config.
- Persisted device selections across app restarts.
- Meeting deletion now removes related audio snapshots and exported notes.

### Changed
- Simplified UI with step-based labels and clearer action text.
- Dynamic speech gate for VAD detection on lower volume sources.

### Fixed
- Mic capture now selects the loudest channel (fixes USB mic channel issues).
- TQDM lock handling during model download and transcription.

## [1.1.25] - 2025-10-12

### Added
- Mix multiple Windows capture sources simultaneously so loopback audio and microphone speech reach the worker together.

### Fixed
- Guarded loopback detection so last_error no longer reports "'NoneType' ... lower" when no input device is selected.
- Fall back to Stereo Mix style loopback inputs when WASAPI loopback streams report PaError -9984.
- Configured mic capture on Windows to request WASAPI shared mode, reducing WDM-KS driver errors.

### Changed
- Overlay caps logger on the capture instance so fallback warnings are always recorded.
- Promote Stereo Mix/Wave Out loopback devices into the playback dropdown to match user expectations.

## [1.1.23] - 2025-10-12

### Fixed
- Replaced remaining self.logger references in start_capture with the capture instance logger to avoid NameError.
- Sanitized dump filenames by lowering (label or "snapshot") so None no longer triggers "NoneType has no attribute lower".

### Changed
- Continue logging loopback candidates for troubleshooting and fall back to microphone only when loopback fails and a mic is selected.

## [1.1.22] - 2025-10-12

### Changed
- Added an explicit "None (no microphone)" option so input capture can be disabled without reloading the app.
- Continue surfacing host API labels alongside device names to make loopback selection clearer.

### Fixed
- Ensure start_capture records last errors for debugging and clears them on successful startups.
- Update capture callbacks to increment frame counters and timestamps reliably.

## [1.1.19] - 2025-10-12

### Changed
- Append host API labels to playback device names so Windows users can pick the WASAPI loopback entry confidently.
- Treat all Windows WASAPI outputs as loopback-capable so loopback devices are auto-selected.

### Fixed
- Ensure start_capture records last errors for debugging and clears them on successful startups.
- Update capture callbacks to increment frame counters and timestamps reliably.

## [1.1.11] - 2025-10-09

### Added
- Automatically migrate any bundled assistant.db into the user data directory so meetings survive reinstalls.
- Surface loopback-capture diagnostics at INFO level for easier troubleshooting.

### Changed
- Tag Windows WASAPI outputs as loopback-capable to highlight the correct playback choices by default.
