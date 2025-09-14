"""
Gated Recurrence Module for Hawk Model
Simplified recurrence implementation without attention
"""

from .hawk_model import HawkModel, GatedRecurrenceBlock

__all__ = [
    "HawkModel",
    "GatedRecurrenceBlock"
]
