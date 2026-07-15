# ZovaAI

ZovaAI is a completely offline-first, local personal AI assistant running on Windows 11. It utilizes open-source models for wake word detection, speech-to-text, text-to-speech, and local LLM intelligence.

---

## Technical Stack (Phase 1)
*   **Audio Capture:** Python `sounddevice` (using system default recording device)
*   **Wake Word:** `openwakeword` ("Hey Jarvis" base model)
*   **Speech-to-Text (STT):** `pywhispercpp` (C++ high-performance Whisper engine bindings)
*   **Text-to-Speech (TTS):** Precompiled C++ `piper` voice synthesizer
*   **Configuration:** `PyYAML` and `.env` overrides
*   **Logging:** Rotating file handler (up to 10MB per file, 5 backup limit)
*   **Testing:** `pytest` unit verification framework

---

## Directory Layout
```
Zova AI/
├── bin/
│   └── piper/                  # Extracted Piper executables & DLLs
├── config/
│   └── config.yaml             # Central configuration (audio rates, paths, URLs)
├── logs/
│   └── assistant.log           # Rotating runtime logs
├── models/
│   ├── piper/                  # TTS ONNX voice files
│   └── whisper/                # STT GGML base model file
├── scripts/
│   ├── __init__.py
│   └── setup_binaries.py       # Binary & model downloader pipeline
├── src/
│   ├── core/                   # Config, DI container, logger, exceptions
│   ├── interfaces/             # Abstract base interface contracts
│   └── main.py                 # Application bootstrapper
├── tests/                      # PyTest unit tests
├── .env                        # Local developer environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # Setup and usage documentation
```

---

## Installation & Setup

### 1. Create Virtual Environment
Create a Python virtual environment and install the required dependencies:
```powershell
python -m venv .venv
.venv\Scripts\activate.ps1
python -m pip install -r requirements.txt
```

### 2. Download Models and Binaries
Run the setup downloader script to download and extract the required AI models and Piper executable files:
```powershell
.venv\Scripts\python -m scripts.setup_binaries
```

#### How the Setup Downloader Works:
1.  **Ensures Directories:** Automatically creates all folders (`bin/piper/`, `models/whisper/`, `models/piper/`, `temp/`).
2.  **Skipping Completed Files:** Reads local file sizes and skips downloads if files are already complete.
3.  **Resuming Partial Downloads:** Checks if the server supports `Range` requests. If a download was interrupted, it resumes from the last byte downloaded instead of starting over.
4.  **Verification:** Validates that each downloaded file matches the expected byte size.
5.  **Zip Extraction:** Unzips the Piper binary payload and deletes temporary ZIP files to optimize disk space.

---

## Running Verification Tests
Ensure that configuration loading, rotating log files, DI Container mappings, and downloader resume mocks are functional by running the test suite:
```powershell
.venv\Scripts\python -m pytest tests/
```
Output should confirm all 12 tests passed successfully.
