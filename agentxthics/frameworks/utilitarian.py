"""
Utilitarian ethical framework for agent decision making.
This framework evaluates actions based on their overall utility to all agents.
"""
from typing import Dict, List, Any, Tuple

from .base import EthicalFramework


class UtilitarianFramework(EthicalFramework):
    """Utilitarian ethics: Maximize overall utility/welfare."""
    
    def __init__(self, weight: float = 0.7):
        super().__init__("Utilitarian", weight)
    
    def evaluate_action(self, agent, action: str, shared_resource) -> float:
        # Calculate expected utility for all agents
        total_agents = len(shared_resource.agents)
        
        # Safety check to avoid division by zero
        if total_agents == 0:
            return 0.5  # Default to neutral score if no agents
        
        if action == "conserve":
            # If all conserve, bonus renewal applies
            if all(a.action == "conserve" for a in shared_resource.agents if a.id != agent.id):
                future_pool = shared_resource.amount - (total_agents * shared_resource.conserve_amount) + shared_resource.bonus_renewal
                expected_utility = future_pool / total_agents
            else:
                future_pool = shared_resource.amount - (total_agents * shared_resource.conserve_amount) + shared_resource.default_renewal
                expected_utility = future_pool / total_agents
                
            # Score higher for conservation when pool is low
            if shared_resource.amount < 30:
                return min(1.0, 0.7 + (30 - shared_resource.amount) / 100)
            
            return min(1.0, expected_utility / 50)
        else:  # consume
            # If agent consumes while others conserve, they get more immediate utility
            conserve_count = sum(1 for a in shared_resource.agents if a.id != agent.id and a.action == "conserve")
            future_pool = shared_resource.amount - (conserve_count * shared_resource.conserve_amount) - ((total_agents - conserve_count) * shared_resource.consume_amount) + shared_resource.default_renewal
            
            # If pool gets dangerously low, consuming is less utilitarian
            if future_pool < 20:
                return max(0.0, future_pool / 100)
            
            # Higher immediate utility for agent, but potentially lower group utility
            individual_utility = shared_resource.consume_amount - shared_resource.conserve_amount
            
            # Balance individual vs. group utility
            if total_agents > 0:
                group_impact = max(0, (future_pool / total_agents) / 50)
                return min(1.0, (individual_utility / 10) * 0.3 + group_impact * 0.7)
            else:
                return min(1.0, (individual_utility / 10) * 0.3)
    
    def get_explanation(self, score: float) -> str:
        if score > 0.7:
            return "maximizes total welfare for all agents"
        elif score > 0.4:
            return "provides reasonable balance of individual and collective utility"
        else:
            return "produces suboptimal utility across the system"
