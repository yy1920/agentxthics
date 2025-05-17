"""
Ethical frameworks for agent decision-making.

This module provides various ethical frameworks that can be used to evaluate
and influence agent decisions in resource management scenarios.
"""

from .base import EthicalFramework
from .utilitarian import UtilitarianFramework
from .deontological import DeontologicalFramework
from .virtue import VirtueEthicsFramework
from .care import CareEthicsFramework
from .justice import JusticeFramework

__all__ = [
    'EthicalFramework',
    'UtilitarianFramework',
    'DeontologicalFramework',
    'VirtueEthicsFramework',
    'CareEthicsFramework',
    'JusticeFramework'
]
