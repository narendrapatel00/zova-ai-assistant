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


def main() -> None:
    """Global main function parsing command line arguments and executing the application."""
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


if __name__ == "__main__":
    main()
