"""
Resource models for agent simulations.

This module provides resource implementations that agents interact with
during simulations, with various properties and renewal mechanisms.
"""

from .base_resource import BaseResource
from .enhanced_resource import EnhancedResource

__all__ = ['BaseResource', 'EnhancedResource']
