"""
Logging System for ZovaAI.
Supports console output and rotating file logs with customizable thresholds and rotation limits.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class LoggerSetup:
    """Setup and helper class for configuring the ZovaAI logging framework."""

    _initialized = False

    @classmethod
    def initialize(
        cls,
        log_level: str = "INFO",
        log_file: Optional[Path] = None,
        max_bytes: int = 10485760,  # 10MB default
        backup_count: int = 5
    ) -> logging.Logger:
        """
        Initializes the logging system with console and rotating file output.
        
        Args:
            log_level: The logging level name (e.g. DEBUG, INFO, WARNING, ERROR).
            log_file: Path to the log file. If none, only console logging is active.
            max_bytes: Maximum size of a log file before rotation.
            backup_count: Number of historical log files to keep.
            
        Returns:
            The configured root Logger instance.
        """
        # Get root logger
        root_logger = logging.getLogger()

        # If already configured, don't re-add handlers (prevents log duplication)
        if cls._initialized:
            return root_logger

        # Set overall level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        root_logger.setLevel(numeric_level)

        # Clear existing handlers if any
        root_logger.handlers.clear()

        # Log format string
        log_format = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
        formatter = logging.Formatter(log_format)

        # 1. Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)

        # 2. Rotating File Handler (Only if log_file is provided)
        if log_file:
            log_file_path = Path(log_file)

            # Ensure containing directories exist
            try:
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                # If directory creation fails, log to console only and print warning
                print(f"Warning: Failed to create log directory '{log_file_path.parent}': {str(e)}")
                cls._initialized = True
                return root_logger

            try:
                file_handler = RotatingFileHandler(
                    filename=log_file_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8"
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(numeric_level)
                root_logger.addHandler(file_handler)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                print(
                    f"Warning: Failed to initialize RotatingFileHandler "
                    f"at '{log_file}': {str(e)}"
                )

        cls._initialized = True
        return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Utility function to retrieve a logger with a given name.
    
    Args:
        name: Name of the logger module.
        
    Returns:
        A logging.Logger instance.
    """
    return logging.getLogger(name)
