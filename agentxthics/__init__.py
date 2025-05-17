"""
AgentXthics: A multi-agent simulation framework for researching ethical decision-making.

This package provides a comprehensive set of tools for simulating agent decision-making
under various ethical frameworks in resource management scenarios.
"""

__version__ = "0.1.0"
__author__ = "AgentX Team"

# Core imports for easier access
from .agents import BaseAgent, EnhancedAgent
from .resources import BaseResource, EnhancedResource
from .frameworks import (
    EthicalFramework,
    UtilitarianFramework,
    DeontologicalFramework,
    VirtueEthicsFramework,
    CareEthicsFramework,
    JusticeFramework
)
