# Changelog

All notable changes to this project will be documented in this file.

## [1.1.22] - 2025-10-12
# (update entries below)

### Changed
- Added an explicit “None (no microphone)” option so input capture can be disabled without reloading the app.
- Continue surfacing host API labels alongside device names to make loopback selection clearer.

### Fixed
- Ensure `start_capture` records last errors for debugging and clears them on successful startups.
- Update capture callbacks to increment frame counters and timestamps reliably.

## [1.1.19] - 2025-10-12

### Changed
- Append host API labels to playback device names so Windows users can pick the WASAPI loopback entry confidently.
- Treat all Windows WASAPI outputs as loopback-capable so loopback devices are auto-selected.

### Fixed
- Ensure `start_capture` records last errors for debugging and clears them on successful startups.
- Update capture callbacks to increment frame counters and timestamps reliably.

## [1.1.11] - 2025-10-09

### Added
- Automatically migrate any bundled `assistant.db` into the user data directory so meetings survive reinstalls.
- Surface loopback-capture diagnostics at `INFO` level for easier troubleshooting.

### Changed
- Tag Windows WASAPI outputs as loopback-capable to highlight the correct playback choices by default.

### Fixed
- Open WASAPI loopback streams with the correct device tuple to prevent capture from stopping immediately after upgrades.
