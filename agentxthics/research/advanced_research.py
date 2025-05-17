"""
Advanced research utilities for AgentXthics.
This module serves as a compatibility layer for the new modular structure.
"""
import os
import simpy
import random
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# Import from modular components
from agents import EnhancedAgent
from resources import EnhancedResource
from research.scenarios import run_simulation
from research.analysis import visualize_resource_metrics, analyze_ethical_frameworks
from research.metrics import calculate_ethical_impact

def run_enhanced_simulation(config=None):
    """
    Run a simulation with enhanced research features.
    This is a compatibility wrapper around the new modular structure.
    
    Args:
        config: Configuration dictionary for the simulation
        
    Returns:
        EnhancedResource object with simulation results
    """
    print("NOTE: Using advanced_research.py is deprecated. Use run_research.py instead.")
    return run_simulation(config)

def run_asymmetric_scenario():
    """
    Run a scenario with asymmetric information.
    Compatibility wrapper for backward compatibility.
    """
    from agentxthics.research.scenarios import run_asymmetric_scenario
    return run_asymmetric_scenario(output_dir='asymmetric_scenario')

def run_vulnerable_scenario():
    """
    Run a scenario with vulnerable populations.
    Compatibility wrapper for backward compatibility.
    """
    from agentxthics.research.scenarios import run_vulnerable_scenario
    return run_vulnerable_scenario(output_dir='vulnerable_scenario')

def run_ethical_framework_comparison():
    """
    Run comparison of different ethical frameworks.
    Compatibility wrapper for backward compatibility.
    """
    from agentxthics.research.scenarios import run_ethical_framework_comparison
    return run_ethical_framework_comparison(output_dir='ethical_frameworks')

def visualize_enhanced_results(resource, output_dir='research_results'):
    """
    Create visualizations for enhanced research metrics.
    Compatibility wrapper around the new modular components.
    """
    metrics_path = visualize_resource_metrics(resource, output_dir)
    ethical_path = analyze_ethical_frameworks(resource, output_dir)
    
    return metrics_path, ethical_path
