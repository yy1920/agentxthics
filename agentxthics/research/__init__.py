"""
Research scenarios and analysis tools.

This module provides predefined simulation scenarios and tools
for analyzing the results of agent-based simulations.
"""

from .scenarios import (
    run_simulation,
    run_asymmetric_scenario,
    run_vulnerable_scenario,
    run_ethical_framework_comparison
)
from .analysis import (
    visualize_resource_metrics,
    analyze_ethical_frameworks,
    compare_ethical_frameworks
)
from .metrics import (
    calculate_fairness,
    calculate_sustainability,
    calculate_welfare,
    calculate_ethical_impact
)
from .run_ethical_frameworks import run_ethical_framework_experiment

__all__ = [
    'run_simulation',
    'run_asymmetric_scenario',
    'run_vulnerable_scenario',
    'run_ethical_framework_comparison',
    'run_ethical_framework_experiment',
    'visualize_resource_metrics',
    'analyze_ethical_frameworks',
    'compare_ethical_frameworks',
    'calculate_fairness',
    'calculate_sustainability',
    'calculate_welfare',
    'calculate_ethical_impact'
]
