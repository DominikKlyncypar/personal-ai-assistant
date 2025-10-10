# Changelog

All notable changes to this project will be documented in this file.

## [1.1.10] - 2025-10-08

### Added
- Automatically migrate any bundled `assistant.db` into the user data directory so meetings survive reinstalls.
- Surface loopback-capture diagnostics at `INFO` level for easier troubleshooting.

### Changed
- Tag Windows WASAPI outputs as loopback-capable to highlight the correct playback choices by default.

### Fixed
- Open WASAPI loopback streams with the correct device tuple to prevent capture from stopping immediately after upgrades.
