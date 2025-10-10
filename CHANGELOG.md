# Changelog

All notable changes to this project will be documented in this file.

## [1.1.7] - 2025-10-08

### Added
- Automatic fail-stop that chunks continuous speech every three minutes and immediately resumes recording with a two-second overlap so no words are lost.
- Configurable overlap setting (`auto_failstop_overlap_s`) exposed in the worker state for future tuning.

### Changed
- Meeting status badge now prefers meeting titles instead of timestamps and keeps metadata trimmed and consistent.
- Meeting creation and deletion flows reuse the normalized meeting labels for clearer toasts and prompts.
- Meeting summaries enforce capitalized, deduplicated bullets across all sections with stricter prompts for the model.
- Heuristic summarizer filters cross-section duplicates and only emits populated sections.

### Fixed
- Prevented duplicate bullets when rendering structured summaries and ensured action items include owners/due dates without duplicating decisions.
- Resolved missing words when speech segments exceed the three minute cap by overlapping chunks.
