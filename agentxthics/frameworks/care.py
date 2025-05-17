"""
Care Ethics framework for agent decision making.
This framework prioritizes relationships and caring for vulnerable parties.
"""
from typing import Dict, List, Any, Tuple

from .base import EthicalFramework


class CareEthicsFramework(EthicalFramework):
    """Care ethics: Prioritize relationships and care for vulnerable parties."""
    
    def __init__(self, weight: float = 0.8):
        super().__init__("Care Ethics", weight)
    
    def evaluate_action(self, agent, action: str, shared_resource) -> float:
        # Identify vulnerable agents (those with special needs or less resources)
        vulnerable_agents = self._identify_vulnerable_agents(shared_resource)
        
        # If this agent is vulnerable, self-care is important
        if agent.id in vulnerable_agents:
            if action == "consume" and shared_resource.amount > 50:
                return 0.8  # It's okay for vulnerable agents to prioritize their needs when resources permit
            elif action == "conserve" and shared_resource.amount < 30:
                return 0.9  # Good to conserve when resources are scarce, even if vulnerable
            else:
                return 0.5
        
        # If there are vulnerable agents, care for them is prioritized
        if vulnerable_agents:
            if action == "conserve":
                # Conservation shows care for vulnerable agents' future needs
                return min(1.0, 0.7 + (len(vulnerable_agents) / len(shared_resource.agents)) * 0.3)
            else:
                # Consumption might harm vulnerable agents' access to resources
                return max(0.0, 0.5 - (len(vulnerable_agents) / len(shared_resource.agents)) * 0.3)
        
        # If no vulnerable agents, evaluate based on general care
        if action == "conserve":
            return 0.7  # Generally caring to conserve
        else:
            return 0.4  # Less caring to consume when no special circumstances apply
    
    def _identify_vulnerable_agents(self, shared_resource) -> List[str]:
        """Identify agents that might be vulnerable or disadvantaged."""
        vulnerable_ids = []
        
        for agent in shared_resource.agents:
            # Check for explicit vulnerability flag
            if hasattr(agent, 'is_vulnerable') and agent.is_vulnerable:
                vulnerable_ids.append(agent.id)
                continue
            
            # Check for asymmetric information (less knowledge is a form of vulnerability)
            if hasattr(agent, 'knowledge_level') and agent.knowledge_level < 0.5:
                vulnerable_ids.append(agent.id)
                continue
            
            # Check for resource disadvantage
            if hasattr(agent, 'resource_need') and agent.resource_need > shared_resource.consume_amount:
                vulnerable_ids.append(agent.id)
                continue
        
        return vulnerable_ids
    
    def get_explanation(self, score: float) -> str:
        if score > 0.7:
            return "demonstrates care and attention to the needs of vulnerable agents"
        elif score > 0.4:
            return "shows moderate concern for relationships and others' needs"
        else:
            return "neglects important caring relationships in the system"
