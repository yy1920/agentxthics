"""
Scenario implementations for AgentXthics research.
These functions create and run different research scenarios with varying parameters.
"""
import os
import simpy
import json
import random
from typing import Dict, Any, Optional

from ..agents import EnhancedAgent
from ..resources import EnhancedResource
from .analysis import visualize_resource_metrics

def run_simulation(config: Dict[str, Any]):
    """
    Run a simulation with the given configuration.
    
    Args:
        config: Configuration dictionary with simulation parameters
        
    Returns:
        EnhancedResource: The resource object after simulation completion
    """
    # Extract common configs
    output_dir = config.get('output_dir', 'simulation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simulation environment
    env = simpy.Environment()
    resource = EnhancedResource(env, config.get('resource', {}))
    
    # Create and register agents
    agent_configs = config.get('agents', [])
    if not agent_configs:
        # Default to 5 agents with different personalities
        agent_configs = [
            {'id': 'A1', 'personality': 'cooperative', 'cooperation_bias': 0.8},
            {'id': 'A2', 'personality': 'cooperative', 'cooperation_bias': 0.7},
            {'id': 'A3', 'personality': 'adaptive', 'cooperation_bias': 0.6},
            {'id': 'A4', 'personality': 'adaptive', 'cooperation_bias': 0.5},
            {'id': 'A5', 'personality': 'competitive', 'cooperation_bias': 0.3}
        ]
    
    agents = []
    for agent_config in agent_configs:
        agent = EnhancedAgent(env, agent_config.get('id', f'A{len(agents)+1}'), resource, agent_config)
        agents.append(agent)
    
    # Run the simulation
    print(f"Starting simulation with {len(agents)} agents for {resource.num_rounds} rounds")
    env.run()
    
    # Save results
    log_paths = resource.save_logs(output_dir)
    
    # Visualize results
    visualize_resource_metrics(resource, output_dir)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        # We need to serialize the config to JSON, but some values might not be serializable
        serializable_config = {k: str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v 
                               for k, v in config.items()}
        json.dump(serializable_config, f, indent=2)
    
    return resource

def run_asymmetric_scenario(output_dir: str = 'asymmetric_scenario'):
    """
    Run a scenario with asymmetric information between agents.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        EnhancedResource: The resource object after simulation completion
    """
    print("\n=== Running Asymmetric Information Scenario ===")
    
    # Set environment variables for this scenario
    os.environ['ENABLE_ASYMMETRIC_INFORMATION'] = 'true'
    os.environ['ENABLE_ETHICAL_FRAMEWORKS'] = 'true'
    
    config = {
        'output_dir': output_dir,
        'resource': {
            'initial_amount': 50,
            'conserve_amount': 5,
            'consume_amount': 10,
            'default_renewal': 15,
            'bonus_renewal': 20,
            'num_rounds': 10
        },
        'agents': [
            # Agent with full information
            {'id': 'A1', 'cooperation_bias': 0.7, 'personality': 'adaptive', 
             'knowledge_level': 1.0, 'custom_prompt': "You have perfect information about the resource pool."},
            
            # Agent with partial information (75%)
            {'id': 'A2', 'cooperation_bias': 0.7, 'personality': 'adaptive', 
             'knowledge_level': 0.75, 'custom_prompt': "You have good but imperfect information about the resource pool."},
            
            # Agent with limited information (50%)
            {'id': 'A3', 'cooperation_bias': 0.7, 'personality': 'adaptive', 
             'knowledge_level': 0.5, 'custom_prompt': "You have limited information about the resource pool."},
             
            # Agent with very limited information (25%)
            {'id': 'A4', 'cooperation_bias': 0.7, 'personality': 'adaptive', 
             'knowledge_level': 0.25, 'custom_prompt': "You have very limited information about the resource pool."}
        ]
    }
    
    return run_simulation(config)

def run_vulnerable_scenario(output_dir: str = 'vulnerable_scenario'):
    """
    Run a scenario with vulnerable populations that have higher resource needs.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        EnhancedResource: The resource object after simulation completion
    """
    print("\n=== Running Vulnerable Populations Scenario ===")
    
    # Set environment variables for this scenario
    os.environ['ENABLE_VULNERABLE_POPULATIONS'] = 'true'
    os.environ['ENABLE_ETHICAL_FRAMEWORKS'] = 'true'
    os.environ['CARE_ETHICS_WEIGHT'] = '0.9'  # Prioritize care ethics
    
    config = {
        'output_dir': output_dir,
        'resource': {
            'initial_amount': 50,
            'conserve_amount': 5,
            'consume_amount': 10,
            'default_renewal': 15,
            'bonus_renewal': 20,
            'num_rounds': 10
        },
        'agents': [
            # Standard agents with normal resource needs
            {'id': 'A1', 'cooperation_bias': 0.7, 'personality': 'cooperative', 
             'resource_need': 10, 'custom_prompt': "You have standard resource needs."},
            {'id': 'A2', 'cooperation_bias': 0.7, 'personality': 'adaptive', 
             'resource_need': 10, 'custom_prompt': "You have standard resource needs."},
             
            # Vulnerable agents with higher resource needs
            {'id': 'A3', 'cooperation_bias': 0.7, 'personality': 'adaptive', 
             'resource_need': 15, 'is_vulnerable': True,
             'custom_prompt': "You are a vulnerable agent with higher resource needs (15 units)."},
            {'id': 'A4', 'cooperation_bias': 0.7, 'personality': 'adaptive', 
             'resource_need': 20, 'is_vulnerable': True,
             'custom_prompt': "You are a highly vulnerable agent with very high resource needs (20 units)."}
        ]
    }
    
    return run_simulation(config)

def run_ethical_framework_comparison(output_dir: str = 'ethical_frameworks'):
    """
    Run multiple scenarios with different ethical frameworks prioritized.
    
    Args:
        output_dir: Base directory to save results
        
    Returns:
        Dict[str, EnhancedResource]: Dictionary of framework names to resource objects
    """
    print("\n=== Running Ethical Framework Comparison ===")
    
    # Define ethical frameworks with different priorities
    frameworks = {
        'utilitarian': {'UTILITARIAN_WEIGHT': '0.9', 'DEONTOLOGICAL_WEIGHT': '0.3', 'VIRTUE_ETHICS_WEIGHT': '0.3', 'CARE_ETHICS_WEIGHT': '0.3', 'JUSTICE_WEIGHT': '0.3'},
        'deontological': {'UTILITARIAN_WEIGHT': '0.3', 'DEONTOLOGICAL_WEIGHT': '0.9', 'VIRTUE_ETHICS_WEIGHT': '0.3', 'CARE_ETHICS_WEIGHT': '0.3', 'JUSTICE_WEIGHT': '0.3'},
        'virtue': {'UTILITARIAN_WEIGHT': '0.3', 'DEONTOLOGICAL_WEIGHT': '0.3', 'VIRTUE_ETHICS_WEIGHT': '0.9', 'CARE_ETHICS_WEIGHT': '0.3', 'JUSTICE_WEIGHT': '0.3'},
        'care': {'UTILITARIAN_WEIGHT': '0.3', 'DEONTOLOGICAL_WEIGHT': '0.3', 'VIRTUE_ETHICS_WEIGHT': '0.3', 'CARE_ETHICS_WEIGHT': '0.9', 'JUSTICE_WEIGHT': '0.3'},
        'justice': {'UTILITARIAN_WEIGHT': '0.3', 'DEONTOLOGICAL_WEIGHT': '0.3', 'VIRTUE_ETHICS_WEIGHT': '0.3', 'CARE_ETHICS_WEIGHT': '0.3', 'JUSTICE_WEIGHT': '0.9'}
    }
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for framework, weights in frameworks.items():
        print(f"\n--- Running {framework.capitalize()} Ethics Scenario ---")
        
        # Set environment variables for this framework
        os.environ['ENABLE_ETHICAL_FRAMEWORKS'] = 'true'
        for key, value in weights.items():
            os.environ[key] = value
        
        # Run simulation for this framework
        framework_dir = os.path.join(output_dir, f"{framework}_scenario")
        config = {
            'output_dir': framework_dir,
            'resource': {
                'initial_amount': 50,
                'conserve_amount': 5,
                'consume_amount': 10,
                'default_renewal': 15,
                'bonus_renewal': 20,
                'num_rounds': 10
            }
        }
        
        results[framework] = run_simulation(config)
    
    # Create comparison visualization
    from .analysis import compare_ethical_frameworks
    compare_ethical_frameworks(results, output_dir)
    
    return results
