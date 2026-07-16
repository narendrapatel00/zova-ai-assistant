# Zova AI Developer Documentation Guide

Welcome to the Zova AI Developer Guide. This document provides an overview of the core architectural patterns, folder structures, and execution models used within the project.

---

## 1. Folder Structure

The project layout divides source code, executable assets, cached model weights, configuration scripts, and integration tests:

```text
Zova AI/
├── bin/                        # Pre-compiled external binaries (e.g., Rhasspy Piper)
├── config/                     # Configuration definitions
│   └── config.yaml             # Settings for Audio, openWakeWord, Whisper, Piper, and LLMs
├── docs/                       # Developer reference manuals and documentation
│   └── developer_guide.md
├── models/                     # Offline neural networks models cache
│   ├── piper/                  # Piper voice profiles (.onnx and JSON files)
│   └── whisper/                # Whisper.cpp GGML files
├── prompts/                    # System prompts configuration
│   └── system.txt              # Active personality prompt injected into the LLM
├── scripts/                    # Command-line executable entry points and utilities
│   ├── demo_assistant.py      # Main voice coordination loops CLI
│   ├── demo_audio.py          # Sounddevice capture diagnostics CLI
│   ├── demo_stt.py            # Whisper.cpp transcription validation CLI
│   ├── demo_tts.py            # Piper vocalization verification CLI
│   ├── demo_wakeword.py       # openWakeWord matching verify CLI
│   └── setup_binaries.py      # Automated model and zip downloads setup pipeline
├── src/                        # Main Zova codebase package
│   ├── assistant/              # Orchestration, Memory, and subscribers
│   ├── audio/                  # PortAudio streams and Session Managers
│   ├── core/                   # Configurations, DI containers, and Exceptions
│   ├── events/                 # Strongly typed events and FIFO queues
│   ├── interfaces/             # Abstract base interface signatures
│   ├── llm/                    # Local Ollama client session managers
│   ├── metrics/                # Latency timer collectors
│   ├── stt/                    # Whisper.cpp transcribers
│   ├── tts/                    # Piper synthesizers and background workers
│   └── wakeword/               # openWakeWord background detectors
└── tests/                      # Pytest unit and integration test suite
```

---

## 2. Dependency Injection (DI)

Zova AI uses a lightweight, thread-safe Dependency Injection (DI) container under `src/core/di.py` to decouple interfaces from concrete implementations:

```python
# Mappings register example:
self.container.register(
    AudioRecorder,
    lambda c: SounddeviceAudioRecorder(c.resolve(Config)),
    singleton=True
)
```

*   **Registry:** Component constructor factories are registered inside `ZovaApp.initialize()` in `src/main.py`.
*   **Resolution:** Callers resolve singletons dynamically at startup (e.g., `orchestrator = container.resolve(AssistantOrchestrator)`), ensuring loose coupling and permitting easy unit mocking.

---

## 3. Event Bus

The communication pattern uses an asynchronous, strongly typed FIFO Event Bus implemented in `src/events/event_bus.py`:

```
Orchestrator ────► publish(Event) ────► queue.Queue ────► Dispatcher Thread ────► ThreadPoolExecutor ────► Subscribers
```

1.  **Strong Typing:** Events inherit from the base `Event` class and are defined as dataclasses (e.g., `STTCompleted`, `LLMChunkReceived`).
2.  **FIFO Guarantee:** Events are queued in a thread-safe `queue.Queue`.
3.  **Concurrency:** A daemon dispatcher thread reads the queue sequentially and dispatches the event tasks to registered subscribers asynchronously via a `ThreadPoolExecutor`. This ensures a slow subscriber never blocks the main orchestrator or audio capture loops.

---

## 4. Streaming Pipeline

The assistant implements low-latency streaming to minimize response times:

```
[User Audio] ──► STT ──► LLM Stream ──► Sentence boundary accumulator ──► TTS background queue ──► Speaker playback
```

1.  **Sentence boundary detection:** Tokens from Ollama are streamed and appended to a character buffer.
2.  **Chunk dispatching:** When the buffer detects a sentence delimiter (`.`, `?`, `!`, `\n`) and has met the minimum character length (`min_chunk_chars: 20`), the chunk is extracted and pushed to the background sequential speaker queue.
3.  **Cancellation:** If a new wake word match is triggered while the assistant is speaking, a shared `CancellationToken` object is flagged cancelled, halting Ollama streams, stopping PortAudio, clearing playing queues, and resetting states instantly.

---

## 5. Audio Pipeline

The hardware recording pipeline remains constantly active to enable seamless offline processing:

```
Microphone ──► AudioRecorder (single InputStream) ──► Shared Queue ──► openWakeWord ──► VAD ──► Whisper.cpp
```

*   **Single InputStream:** The microphone stream is never closed or reopened between cycles. It streams 16kHz mono PCM frames continuously.
*   **Shared Ring Buffers:** Audio frames are broadcasted to a sliding buffer checked by openWakeWord. On wake word detection, frames continue routing to a Voice Activity Detection (VAD) audio accumulator until a configurable silence period finishes, writing the segment to a single WAV path.

---

## 6. Thread Model

Tasks are distributed across specialized threads to prevent UI freezes or sound card glitches:

| Thread Name | Daemon? | Responsibility |
| :--- | :--- | :--- |
| `Main Thread` | No | CLI execution, boot, and exit loops |
| `sounddevice` PortAudio C Thread | Yes | Low-level microphone capture callbacks |
| `WakeWordListeningService` | Yes | Background openWakeWord inference runs |
| `EventBusDispatcher` | Yes | Asynchronous events FIFO queue parser |
| `event_bus_worker_*` Pool | Yes | Running subscriber callbacks (Logging, Metrics) |
| `TTSStreamingWorker` | Yes | Background sequential sentence Piper synthesis and playback |
