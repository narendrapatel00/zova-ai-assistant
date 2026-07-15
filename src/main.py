"""
Main Entry Point for ZovaAI.
Initializes configuration, logging, dependency injection, and components.
Sets up the scaffolding for Phase 1 voice loops.
"""

import sys
from typing import Optional

from src.core.config import Config
from src.core.di import DIContainer
from src.core.logger import LoggerSetup, get_logger
from src.core.exceptions import ZovaException
from src.interfaces.audio_recorder import AudioRecorder
from src.interfaces.wake_word_detector import WakeWordDetector
from src.interfaces.speech_recognizer import SpeechRecognizer
from src.interfaces.speech_synthesizer import SpeechSynthesizer
from src.audio.recorder import SounddeviceAudioRecorder
from src.wakeword.interfaces import WakeWordService
from src.wakeword.engine import OpenWakeWordDetector
from src.wakeword.service import WakeWordListeningService
from src.stt.recognizer import WhisperSpeechRecognizer
from src.stt.service import SpeechRecognitionService
from src.tts.synthesizer import PiperSpeechSynthesizer
from src.tts.service import SpeechSynthesisService
from src.llm.ollama_client import LLMClient, OllamaLLMClient
from src.llm.service import LLMService
from src.assistant.orchestrator import AssistantOrchestrator

logger = get_logger("main")


class ZovaApp:
    """Core Application Wrapper managing initialization and lifecycle."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the application, container, and configuration.
        
        Args:
            config_path: Custom path to config.yaml (optional).
        """
        self.config: Optional[Config] = None
        self.container = DIContainer()
        self.config_path = config_path

    def initialize(self) -> None:
        """
        Runs the core application bootstrapping sequence:
        1. Loads settings from configuration file.
        2. Configures console and rotating file logger handlers.
        3. Configures dependency injection registrations.
        
        Raises:
            ZovaException: If initialization sequence fails.
        """
        try:
            # 1. Load configuration
            self.config = Config(self.config_path)
            
            # 2. Setup logging based on configuration
            LoggerSetup.initialize(
                log_level=self.config.logging.level,
                log_file=self.config.logging.log_file,
                max_bytes=self.config.logging.max_bytes,
                backup_count=self.config.logging.backup_count
            )
            
            logger.info("=========================================")
            logger.info("Initializing ZovaAI Core Application...")
            logger.info("Environment: %s", self.config.app.env)
            logger.info("Version: %s", self.config.app.version)
            
            # 3. Register core configuration with DI Container
            self.container.register(Config, self.config, singleton=True)
            
            # 4. Register concrete audio recorder implementation
            self.container.register(
                AudioRecorder,
                lambda c: SounddeviceAudioRecorder(c.resolve(Config)),
                singleton=True
            )

            # 5. Register concrete wake word detector
            self.container.register(
                WakeWordDetector,
                lambda c: OpenWakeWordDetector(c.resolve(Config)),
                singleton=True
            )

            # 6. Register concrete wake word listening service
            self.container.register(
                WakeWordService,
                lambda c: WakeWordListeningService(
                    c.resolve(Config),
                    c.resolve(AudioRecorder),
                    c.resolve(WakeWordDetector)
                ),
                singleton=True
            )

            # 7. Register concrete speech recognizer
            self.container.register(
                SpeechRecognizer,
                lambda c: WhisperSpeechRecognizer(c.resolve(Config)),
                singleton=True
            )

            # 8. Register speech recognition service
            self.container.register(
                SpeechRecognitionService,
                lambda c: SpeechRecognitionService(c.resolve(SpeechRecognizer)),
                singleton=True
            )

            # 9. Register concrete speech synthesizer
            self.container.register(
                SpeechSynthesizer,
                lambda c: PiperSpeechSynthesizer(c.resolve(Config)),
                singleton=True
            )

            # 10. Register speech synthesis service
            self.container.register(
                SpeechSynthesisService,
                lambda c: SpeechSynthesisService(c.resolve(SpeechSynthesizer)),
                singleton=True
            )

            # 11. Register LLM Client
            self.container.register(
                LLMClient,
                lambda c: OllamaLLMClient(c.resolve(Config)),
                singleton=True
            )

            # 12. Register LLM Service
            self.container.register(
                LLMService,
                lambda c: LLMService(c.resolve(LLMClient)),
                singleton=True
            )

            # 13. Register Assistant Orchestrator
            self.container.register(
                AssistantOrchestrator,
                lambda c: AssistantOrchestrator(
                    c.resolve(Config),
                    c.resolve(AudioRecorder),
                    c.resolve(WakeWordService),
                    c.resolve(SpeechRecognizer),
                    c.resolve(LLMService),
                    c.resolve(SpeechSynthesisService)
                ),
                singleton=True
            )
            
            logger.info("Dependency Injection Container initialized.")
            logger.info("Scaffolding initialization complete. Ready for engines.")
            logger.info("=========================================")
            
        except ZovaException as ze:
            # Re-raise known ZovaExceptions
            raise ze
        # pylint: disable=broad-exception-caught
        except Exception as e:
            # Wrap standard exceptions
            raise ZovaException(f"Unhandled bootstrap error: {str(e)}") from e

    def run(self) -> None:
        """
        Runs the application voice control loop.
        To be implemented in Milestone 2.
        """
        logger.info(
            "ZovaAI running in idle mode. Voice loops will be implemented in Milestone 2."
        )

    def close(self) -> None:
        """Gracefully stops all background threads and releases audio card hardware locks."""
        logger.info("Shutting down ZovaAI application...")
        
        # Stop background wake-word listener thread
        try:
            ww_service = self.container.resolve(WakeWordService)
            if ww_service.is_running():
                ww_service.stop()
        # pylint: disable=broad-exception-caught
        except Exception:
            pass

        # Stop background audio recorder stream
        try:
            recorder = self.container.resolve(AudioRecorder)
            recorder.close()
        # pylint: disable=broad-exception-caught
        except Exception:
            pass

        # Close speech recognizer context
        try:
            recognizer = self.container.resolve(SpeechRecognizer)
            recognizer.close()
        # pylint: disable=broad-exception-caught
        except Exception:
            pass

        # Close speech synthesizer context
        try:
            synthesizer = self.container.resolve(SpeechSynthesizer)
            synthesizer.close()
        # pylint: disable=broad-exception-caught
        except Exception:
            pass

        # Close LLM Client Session
        try:
            llm_client = self.container.resolve(LLMClient)
            llm_client.close()
        # pylint: disable=broad-exception-caught
        except Exception:
            pass

        # Close Assistant Orchestrator
        try:
            orchestrator = self.container.resolve(AssistantOrchestrator)
            if orchestrator.is_running():
                orchestrator.stop()
        # pylint: disable=broad-exception-caught
        except Exception:
            pass

        logger.info("ZovaAI shutdown complete.")


def main() -> None:
    """Global main function parsing command line arguments and executing the application."""
    app = None
    try:
        app = ZovaApp()
        app.initialize()
        app.run()
    except ZovaException as ze:
        print(f"CRITICAL BOOTSTRAP ERROR: {ze.message}", file=sys.stderr)
        sys.exit(1)
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"CRITICAL SYSTEM CRASH: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if app:
            app.close()


if __name__ == "__main__":
    main()
