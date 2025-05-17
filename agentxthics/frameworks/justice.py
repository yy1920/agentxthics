"""
Justice Ethics framework for agent decision making.
This framework evaluates actions based on fair distribution of benefits and burdens.
"""
import math
from typing import Dict, List, Any, Tuple

from .base import EthicalFramework


class JusticeFramework(EthicalFramework):
    """Justice ethics: Distribute benefits and burdens fairly."""
    
    def __init__(self, weight: float = 0.7):
        super().__init__("Justice", weight)
    
    def evaluate_action(self, agent, action: str, shared_resource) -> float:
        # Calculate current resource distribution
        current_distribution = self._get_resource_distribution(shared_resource)
        
        # Calculate Gini coefficient as a measure of inequality
        gini = self._calculate_gini(current_distribution)
        
        # Estimate how the action affects future inequality
        if action == "conserve":
            # Conservation tends to maintain current distribution
            future_impact = 0.0
        else:  # consume
            # Safety check for empty distribution
            distribution_sum = sum(current_distribution.values())
            if not current_distribution or distribution_sum == 0 or not shared_resource.agents:
                future_impact = 0.0
            else:
                # Consumption by those with more resources increases inequality
                agent_current_share = current_distribution.get(agent.id, 0) / distribution_sum
                if agent_current_share > 1.0 / len(shared_resource.agents):
                    # Agent already has above-average resources
                    future_impact = 0.1  # Increase inequality
                else:
                    # Agent has below-average resources
                    future_impact = -0.05  # Slight decrease in inequality
        
        # Evaluate based on inequality and action's impact
        if gini < 0.2:  # Low inequality
            if action == "conserve":
                return 0.7  # Good to conserve when distribution is already fair
            else:
                # Can be acceptable to consume if resources are abundant and inequality is low
                return 0.5 if shared_resource.amount > 50 else 0.4
        else:  # Higher inequality
            if action == "conserve":
                return 0.8  # Very good to conserve when inequality exists
            else:
                # Consuming when inequality exists depends on agent's position
                if not shared_resource.agents:
                    return 0.5  # Neutral score if no agents to compare
                elif agent_current_share < 1.0 / len(shared_resource.agents):
                    return 0.6  # More just for disadvantaged to consume
                else:
                    return 0.3  # Less just for advantaged to consume
    
    def _get_resource_distribution(self, shared_resource) -> Dict[str, float]:
        """Estimate the current resource distribution among agents."""
        distribution = {}
        
        # Safety check for no agents
        if not shared_resource.agents:
            return {}
        
        # Start with equal distribution as a baseline
        base_share = shared_resource.amount / len(shared_resource.agents)
        
        for agent in shared_resource.agents:
            # Adjust based on past consumption patterns
            if hasattr(agent, 'consumption_history') and agent.consumption_history:
                total_consumption = sum(agent.consumption_history)
                avg_consumption = total_consumption / len(agent.consumption_history)
                
                # Agents who consumed more in the past likely have more resources now
                distribution[agent.id] = base_share * (avg_consumption / shared_resource.consume_amount)
            else:
                distribution[agent.id] = base_share
        
        return distribution
    
    def _calculate_gini(self, distribution: Dict[str, float]) -> float:
        """Calculate Gini coefficient as a measure of inequality."""
        if not distribution:
            return 0.0
        
        values = list(distribution.values())
        if not values or sum(values) == 0:
            return 0.0
        
        # Sort values
        values.sort()
        n = len(values)
        
        # Calculate Gini coefficient
        cumsum = 0
        for i, value in enumerate(values):
            cumsum += (2 * i - n + 1) * value
        
        return cumsum / (n * n * sum(values) / n)
    
    def get_explanation(self, score: float) -> str:
        if score > 0.7:
            return "promotes a fair distribution of resources among all agents"
        elif score > 0.4:
            return "maintains reasonable fairness in resource allocation"
        else:
            return "creates or perpetuates unjust resource distribution"
