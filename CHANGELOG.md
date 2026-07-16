# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-rc1] - 2026-07-16

### Added
- **Asynchronous FIFO Event Bus:** Created a strongly typed async event bus under `src/events/` running subscriber callbacks asynchronously on a ThreadPoolExecutor.
- **Audio Session Manager:** Added `AudioSessionManager` with an Enum-based state machine (`IDLE`, `LISTENING`, `RECORDING`, `PROCESSING`, `SPEAKING`, `SHUTTING_DOWN`) for pipeline state tracking.
- **Sentence-based LLM Streaming:** Implemented Ollama token streaming and sentence boundary parser to synthesize speech sentence-by-sentence.
- **Sequential Streaming TTS Playback:** Introduced `StreamingTTSWorker` sequentially vocalizing sentence buffers using a background worker thread.
- **Conversation Cancellation:** Implemented shared `CancellationToken` checks across the orchestrator, generator loop, and TTS player to halt playbacks instantly on wake word match.
- **Centralized Metrics Collector:** Created `MetricsCollector` measuring wake-word latency, recording duration, STT latency, LLM time-to-first-token, generation rates (TPS), and end-to-end processing delays.
- **Offline TTS Integration:** Created `PiperSpeechSynthesizer` invoking Rhasspy's Piper engine locally.
- **Offline STT Integration:** Created `WhisperSpeechRecognizer` wrapping pywhispercpp (Whisper.cpp) for local WAV transcription.
- **Wake Word Detection:** Integrated `OpenWakeWordDetector` implementing OpenWakeWord background matching.
- **Audio Recording:** Added `SounddeviceAudioRecorder` with a Voice Activity Detection (VAD) buffer and silence threshold timeouts.
- **Dependency Injection:** Integrated `DIContainer` for component mapping and lifecycle cleanups.

### Changed
- Refactored `AssistantOrchestrator` to coordinate event-driven transitions, metric updates, and speech cancellations.
- Updated `scripts/demo_assistant.py` to register console handlers directly through event bus subscriptions.

### Fixed
- Fixed trailing whitespace lints across all Python packages.
- Resolved type compilation errors under `mypy`.
- Cleaned unused typing and exception imports in `orchestrator.py`.
