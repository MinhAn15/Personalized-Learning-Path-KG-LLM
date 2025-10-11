"""Top-level shim package to expose backend.src as src.

This allows code that imports `src` to continue working without moving files.
"""
from importlib import import_module

try:
    # Import the config module inside backend.src where Config is defined
    _backend_config = import_module("backend.src.config")
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "backend.src.config not found. Ensure you're running from the repository root and that 'backend/src/config.py' exists"
    ) from e

# Expose Config class from backend.src.config
Config = getattr(_backend_config, "Config", None)
if Config is None:
    raise ImportError("Config class not found in backend.src.config")

__all__ = ["Config"]
