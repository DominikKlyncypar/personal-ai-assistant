# Changelog

All notable changes to this project will be documented in this file.

## [1.1.11] - 2025-10-09

### Fixed
- Updated mic and WASAPI loopback callbacks to update frame counters and timestamps, preventing false "frozen" detection and worker exits.
- Improved loopback logging to flag callback status warnings and surface failures for troubleshooting.

### Changed
- Treat all Windows WASAPI outputs as loopback-capable so the correct playback option is selected automatically.

## [1.1.10] - 2025-10-08

### Added
- Automatically migrate any bundled `assistant.db` into the user data directory so meetings survive reinstalls.
- Surface loopback-capture diagnostics at `INFO` level for easier troubleshooting.

### Changed
- Tag Windows WASAPI outputs as loopback-capable to highlight the correct playback choices by default.

### Fixed
- Open WASAPI loopback streams with the correct device tuple to prevent capture from stopping immediately after upgrades.
