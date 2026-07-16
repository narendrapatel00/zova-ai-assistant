# ZovaAI

ZovaAI is a completely offline-first, local personal voice assistant running on Windows. It utilizes open-source models for wake word detection, speech-to-text, local LLM brain logic, and high-quality text-to-speech vocalization.

---

## 1. Architecture Overview

ZovaAI uses a modular, event-driven, decoupled pipeline:

```text
Microphone ──► sounddevice (Ring Buffer) ──► openWakeWord ──► VAD ──► Whisper.cpp (STT)
                                                                           │
                                                                           ▼
[User Hear Speech] ◄── sounddevice ◄── Sequential TTS Worker ◄── Ollama (Qwen Stream)
```

-   **Continuous Audio Loop:** Captures 16kHz PCM audio on a single continuous stream.
-   **Wake Word Match:** openWakeWord detects "Hey Jarvis" locally on CPU.
-   **Command Recording:** VAD matches silence thresholds to stop command recording automatically.
-   **Asynchronous Event Bus:** Coordinates pipeline state changes asynchronously, sending telemetry to Logging and Metrics subscribers.
-   **Low-Latency Speech Synthesizer:** Sentences are streamed chunk-by-sentence to a sequential Piper synthesiser background thread for playback under 1.0 second.
-   **Interruptible Playback:** If a wake word is matched during playback, a cancellation token cancels all background threads and stops audio instantly.

---

## 2. Technical Stack

*   **Audio Capture:** Python `sounddevice` (using system default recording device)
*   **Wake Word:** `openwakeword` ("Hey Jarvis" model)
*   **Speech-to-Text (STT):** `pywhispercpp` (C++ high-performance Whisper base model bindings)
*   **Brain LLM:** local Ollama instance (running `qwen2.5:latest` or `qwen3`)
*   **Text-to-Speech (TTS):** Rhasspy C++ `piper` voice synthesizer
*   **Configuration:** `PyYAML` and `.env` overrides
*   **Testing:** `pytest` unit verification framework

---

## 3. Installation & Setup

### Prerequisites
*   Python 3.10+ (Windows 11 recommended)
*   Ollama for Windows installed and running locally ([Download Ollama](https://ollama.com))

### 1. Create Virtual Environment
Create a Python virtual environment and install the required dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Download Models and Binaries
Run the setup downloader script to download and extract the required AI models (Whisper GGML model, Piper ONNX model, and Piper executable files):
```powershell
.venv\Scripts\python -m scripts.setup_binaries
```

### 3. Start Local Ollama Server
Pull the default Qwen model locally using Ollama:
```powershell
ollama pull qwen2.5:latest
```
Ensure Ollama is running in the background (typically accessible at `http://localhost:11434`).

---

## 4. Running the Assistant

Execute the orchestrator voice control loop:
```powershell
.venv\Scripts\python -m scripts.demo_assistant
```
1.  Wait for the console log to show `[Idle] Ready. Listening for wake word: 'Jarvis'...`.
2.  Say **"Hey Jarvis"** or **"Jarvis"** near your microphone.
3.  Upon hearing the activation chime, speak your command (e.g., *"What is the distance between the Earth and the Moon?"*).
4.  The assistant will stream the response from Ollama and read it back sentence-by-sentence.
5.  Press `Ctrl+C` to stop the assistant and close audio drivers cleanly.

---

## 5. Running Tests

Run the test suite to verify code correctness:
```powershell
# Run all unit and integration tests
.venv\Scripts\python -m pytest tests/

# Run static type checking
.venv\Scripts\mypy src/

# Run code quality linter checks
.venv\Scripts\pylint src/ scripts/ --disable=C0114,C0115,C0116,R0903,W0212
```

---

## 6. Troubleshooting

*   **Audio Device Disconnected or PortAudio Errors:**
    Open `config/config.yaml` and set `device_index` under the `audio` namespace to the specific index of your system input card (instead of `null`). Run `python -m scripts.demo_audio` to list available audio hardware.
*   **missing tflite-runtime warnings:**
    If openwakeword complains about `tflite-runtime` missing, run `pip install tflite-runtime` or run the setup downloader script `python -m scripts.setup_binaries` which downloads cached ONNX models to bypass tflite.
*   **Ollama Connection Refused:**
    Ensure Ollama is running on your machine by visiting `http://localhost:11434` in your browser. Verify the `model` key in `config/config.yaml` matches the model tag returned by running `ollama list`.

---

## 7. Project Roadmap

*   **Phase 1: Audio Core & Pipeline Scaffolding (Completed)**
    *   WAV capturing, openWakeWord matching, local Whisper.cpp STT, local Piper TTS.
*   **Phase 2: Assistant Orchestrator & Low-Latency Streaming (Completed)**
    *   Central Orchestrator, Ollama Client Session, Decoupled Asynchronous FIFO Event Bus, SessionState transition managers, shared CancellationToken, and sentence chunk streaming.
*   **Phase 3: Persistent Memory (Up Next)**
    *   SQLite database integration, conversation historical context pruning, user profile storage, and memory indexing.
*   **Phase 4: Plugins & PC Control**
    *   Local weather, shell automation commands, web browser searching, and desktop automation tools.
