"""
Central Performance Metrics Subsystem for ZovaAI.
Consolidates latency, generation time, and execution duration measurements
across all voice assistant pipelines in a thread-safe coordinator.
"""

import time
import threading
from typing import Dict, Optional

from src.core.logger import get_logger

logger = get_logger("performance_metrics")


class MetricsCollector:
    """Thread-safe performance metrics accumulator."""

    def __init__(self) -> None:
        """Initializes the metrics collector."""
        self._timers: Dict[str, float] = {}
        self._metrics: Dict[str, float] = {}
        self._lock = threading.Lock()

    def start_timer(self, name: str) -> None:
        """
        Starts a timer for a metric.
        
        Args:
            name: The metric name identifier.
        """
        with self._lock:
            self._timers[name] = time.perf_counter()
            logger.debug("Performance timer started: %s", name)

    def stop_timer(self, name: str) -> Optional[float]:
        """
        Stops a timer for a metric, computing and storing the duration.
        
        Args:
            name: The metric name identifier.
            
        Returns:
            Optional[float]: Calculated duration in seconds, or None if timer not started.
        """
        stop_time = time.perf_counter()
        with self._lock:
            start_time = self._timers.pop(name, None)
            if start_time is None:
                logger.warning("Attempted to stop timer '%s' which was never started.", name)
                return None
            
            duration = stop_time - start_time
            self._metrics[name] = duration
            logger.debug("Performance timer stopped: %s (Duration: %.4fs)", name, duration)
            return duration

    def set_value(self, name: str, value: float) -> None:
        """
        Manually sets a metric to a specific numeric value.
        
        Args:
            name: Metric name identifier.
            value: Numerical metric value to store.
        """
        with self._lock:
            self._metrics[name] = value

    def get_value(self, name: str) -> Optional[float]:
        """
        Retrieves a stored metric value.
        
        Args:
            name: Metric identifier.
            
        Returns:
            Optional[float]: Stored value, or None if not found.
        """
        with self._lock:
            return self._metrics.get(name)

    def get_all(self) -> Dict[str, float]:
        """
        Retrieves a copy of all accumulated metrics.
        
        Returns:
            Dict[str, float]: Stored key-value metrics dictionary.
        """
        with self._lock:
            return self._metrics.copy()

    def clear(self) -> None:
        """Wipes all timers and stored metric values."""
        with self._lock:
            self._timers.clear()
            self._metrics.clear()
            logger.info("Central performance metrics collector cleared.")

    def log_summary(self) -> None:
        """Writes a formatted key-value block of all accumulated metrics to logs."""
        metrics = self.get_all()
        if not metrics:
            logger.info("No performance metrics captured in this session.")
            return

        logger.info("=========================================")
        logger.info("   ZOVA AI PERFORMANCE BENCHMARKS")
        logger.info("=========================================")
        for key, value in sorted(metrics.items()):
            if "tps" in key or "rate" in key or "count" in key:
                logger.info("  %-30s: %.2f", key, value)
            else:
                logger.info("  %-30s: %.4fs", key, value)
        logger.info("=========================================")
