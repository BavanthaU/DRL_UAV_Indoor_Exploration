"""Mapping interfaces, fallback backend, and pure grid utilities."""

from .base import MappingBackend
from .existing_fallback_backend import MapFromExistingOccupancyBackend

__all__ = ["MappingBackend", "MapFromExistingOccupancyBackend"]
