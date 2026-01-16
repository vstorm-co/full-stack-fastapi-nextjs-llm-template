"""FastAPI Project Generator with Logfire observability."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fastapi-fullstack")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Development/editable install fallback
