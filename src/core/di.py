"""
Dependency Injection (DI) Container for ZovaAI.
Allows decoupling interfaces from concrete implementations, facilitating clean testing
and modular plugin upgrades.
"""

from typing import Any, Dict, Type, Optional
from src.core.exceptions import DependencyInjectionError
from src.core.logger import get_logger

logger = get_logger(__name__)


class DIContainer:
    """A lightweight Dependency Injection container supporting Singleton/Factory bindings."""

    _instance: Optional['DIContainer'] = None
    _registry: Dict[Type[Any], Dict[str, Any]]
    _singletons: Dict[Type[Any], Any]

    def __new__(cls, *args: Any, **kwargs: Any) -> 'DIContainer':
        """Ensure DIContainer behaves as a singleton container for the lifecycle of the app."""
        if not cls._instance:
            cls._instance = super(DIContainer, cls).__new__(cls, *args, **kwargs)
            cls._instance._registry = {}
            cls._instance._singletons = {}
        return cls._instance

    def register(self, interface: Type[Any], implementation: Any, singleton: bool = True) -> None:
        """
        Registers an implementation for a given interface type.

        Args:
            interface: The abstract base class or interface type.
            implementation: Either an instance of the class, or a callable factory function.
            singleton: If True, registrations of callables are evaluated once and cached.

        Raises:
            DependencyInjectionError: If arguments are invalid.
        """
        if interface is None:
            raise DependencyInjectionError("Cannot register None as an interface type.")

        self._registry[interface] = {
            "impl": implementation,
            "singleton": singleton
        }

        # If it's a pre-constructed object/instance and not a callable factory,
        # we store it directly in singletons.
        if singleton and not callable(implementation):
            self._singletons[interface] = implementation

        logger.debug(
            "Registered %s -> %s (singleton=%s)",
            interface.__name__,
            implementation,
            singleton
        )

    def resolve(self, interface: Type[Any]) -> Any:
        """
        Resolves and returns the implementation registered for the given interface.

        Args:
            interface: The interface type to resolve.

        Returns:
            The resolved instance.

        Raises:
            DependencyInjectionError: If the dependency is not registered or fails to resolve.
        """
        if interface not in self._registry:
            raise DependencyInjectionError(
                f"Dependency for interface '{interface.__name__}' is not registered."
            )

        entry = self._registry[interface]
        impl = entry["impl"]
        is_singleton = entry["singleton"]

        if is_singleton:
            if interface in self._singletons:
                return self._singletons[interface]

            # If the singleton has not been constructed yet, invoke the factory
            if callable(impl):
                try:
                    logger.debug("Instantiating singleton dependency: %s", interface.__name__)
                    instance = impl(self)
                    self._singletons[interface] = instance
                    return instance
                except Exception as e:
                    raise DependencyInjectionError(
                        f"Failed to instantiate singleton dependency "
                        f"'{interface.__name__}': {str(e)}"
                    ) from e
            else:
                # Fallback if somehow not cached
                self._singletons[interface] = impl
                return impl
        else:
            # Factory pattern: construct new instance every time
            if callable(impl):
                try:
                    logger.debug("Instantiating factory dependency: %s", interface.__name__)
                    return impl(self)
                except Exception as e:
                    raise DependencyInjectionError(
                        f"Failed to instantiate factory dependency "
                        f"'{interface.__name__}': {str(e)}"
                    ) from e
            else:
                return impl

    def clear(self) -> None:
        """Clears all registered dependencies and singleton instances. Useful for testing."""
        self._registry.clear()
        self._singletons.clear()
        logger.debug("DIContainer registry cleared.")
