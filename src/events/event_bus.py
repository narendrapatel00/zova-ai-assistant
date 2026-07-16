"""
FIFO Asynchronous Event Bus for ZovaAI.
Supports subscribing and publishing strongly typed events, ensuring
FIFO ordering via queues and thread-pool asynchronous dispatching.
"""

import queue
import threading
from typing import Callable, Type, Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from src.core.logger import get_logger
from src.events.events import Event

logger = get_logger("event_bus")


class EventBus:
    """Thread-safe, FIFO event bus executing subscribers asynchronously."""

    def __init__(self, max_workers: int = 5):
        """
        Initializes the event bus.
        
        Args:
            max_workers: Size of the thread pool used for async callbacks.
        """
        self._listeners: Dict[Type[Event], List[Callable[[Any], None]]] = {}
        self._queue: queue.Queue = queue.Queue()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="event_bus_worker"
        )
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Starts the background event dispatch thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            
        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            name="EventBusDispatcher",
            daemon=True
        )
        self._dispatcher_thread.start()
        logger.info("Event Bus dispatcher thread started.")

    def stop(self) -> None:
        """Stops the dispatcher and shuts down the thread pool."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            
        # Put sentinel to unblock queue.get()
        self._queue.put(None)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        logger.info("Event Bus thread pool executor shutdown.")

    def join(self, timeout: float = 1.0) -> None:
        """
        Waits for the dispatcher thread to terminate.
        
        Args:
            timeout: Maximum seconds to block.
        """
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=timeout)
            logger.info("Event Bus dispatcher thread joined.")

    def is_running(self) -> bool:
        """
        Checks if dispatcher thread is active.
        
        Returns:
            bool: True if active, False otherwise.
        """
        with self._lock:
            return self._running and (
                self._dispatcher_thread is not None and
                self._dispatcher_thread.is_alive()
            )

    def subscribe(self, event_type: Type[Event], callback: Callable[[Any], None]) -> None:
        """
        Subscribes a callback to an event type.
        
        Args:
            event_type: Dataclass type inheriting from Event.
            callback: Function to invoke when event is matched.
        """
        with self._lock:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            if callback not in self._listeners[event_type]:
                self._listeners[event_type].append(callback)
                logger.debug(
                    "Subscribed %s to %s",
                    callback.__name__ if hasattr(callback, "__name__") else str(callback),
                    event_type.__name__
                )

    def unsubscribe(self, event_type: Type[Event], callback: Callable[[Any], None]) -> None:
        """
        Unsubscribes a callback from an event type.
        
        Args:
            event_type: Event type class.
            callback: Registered callback function.
        """
        with self._lock:
            if event_type in self._listeners:
                if callback in self._listeners[event_type]:
                    self._listeners[event_type].remove(callback)
                    logger.debug(
                        "Unsubscribed %s from %s",
                        callback.__name__ if hasattr(callback, "__name__") else str(callback),
                        event_type.__name__
                    )

    def publish(self, event: Event) -> None:
        """
        Pushes an event into the FIFO queue for dispatching.
        
        Args:
            event: Event object instance.
        """
        if not self.is_running():
            logger.warning("Event published but EventBus is not running: %s", type(event).__name__)
        self._queue.put(event)

    def _dispatch_loop(self) -> None:
        """Background loop reading from queue and dispatching to thread pool."""
        while True:
            try:
                event = self._queue.get()
                if event is None:
                    # Shutdown sentinel received
                    self._queue.task_done()
                    break
                
                event_type = type(event)
                with self._lock:
                    listeners = self._listeners.get(event_type, []).copy()
                
                # Submit each listener callback to run asynchronously in thread pool
                for listener in listeners:
                    self._executor.submit(self._safe_execute, listener, event)
                
                self._queue.task_done()
            except Exception as e:
                logger.error("Error in EventBus dispatch loop: %s", e)

    def _safe_execute(self, listener: Callable[[Any], None], event: Event) -> None:
        """Executes a subscriber callback catching and logging any exceptions."""
        try:
            listener(event)
        except Exception as e:
            logger.error(
                "Error executing listener callback %s for event %s: %s",
                listener.__name__ if hasattr(listener, "__name__") else str(listener),
                type(event).__name__,
                e
            )
