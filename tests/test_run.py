"""Simple test script for running AgentXthics with a basic configuration."""
import os
import simpy
from agents.enhanced_agent import EnhancedAgent
from resources.enhanced_resource import EnhancedResource

# Create basic configuration
config = {
    'output_dir': 'test_output',
    'resource': {
        'initial_amount': 50,
        'conserve_amount': 5,
        'consume_amount': 10,
        'default_renewal': 15,
        'bonus_renewal': 20,
        'num_rounds': 3  # Reduced for testing
    },
    'agents': [
        {'id': 'A1', 'personality': 'cooperative', 'cooperation_bias': 0.8},
        {'id': 'A2', 'personality': 'adaptive', 'cooperation_bias': 0.5}
    ]
}

def run_test():
    """Run a simple test simulation."""
    print("Starting test simulation...")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create simulation environment
    env = simpy.Environment()
    resource = EnhancedResource(env, config.get('resource', {}))
    
    # Create agents
    agents = []
    for agent_config in config['agents']:
        agent = EnhancedAgent(env, agent_config.get('id'), resource, agent_config)
        agents.append(agent)
    
    # Run simulation
    env.run()
    
    # Save logs
    resource.save_logs(config['output_dir'])
    
    print(f"Test simulation complete. Results saved to {config['output_dir']}")
    return resource

if __name__ == "__main__":
    run_test()
