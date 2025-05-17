"""
Virtue Ethics framework for agent decision making.
This framework evaluates actions based on virtuous character traits.
"""
from typing import Dict, List, Any, Tuple, Optional

from .base import EthicalFramework


class VirtueEthicsFramework(EthicalFramework):
    """Virtue ethics: Act according to virtuous character traits."""
    
    def __init__(self, weight: float = 0.6):
        super().__init__("Virtue Ethics", weight)
        self.virtues = {
            "cooperation": 0.0,
            "moderation": 0.0,
            "wisdom": 0.0,
            "fairness": 0.0
        }
    
    def evaluate_action(self, agent, action: str, shared_resource) -> float:
        # Update virtue assessments based on history
        self._update_virtues(agent, shared_resource)
        
        # Evaluate action against virtues
        if action == "conserve":
            # Conservation demonstrates cooperation
            cooperation_score = 0.8
            
            # Moderation depends on resource state
            if shared_resource.amount > 80:
                # With abundant resources, moderation might mean some consumption is fine
                moderation_score = 0.5
            else:
                moderation_score = 0.7
            
            # Wisdom considers resource trends
            if self._is_resource_declining(shared_resource):
                wisdom_score = 0.9  # Wise to conserve when resources are declining
            else:
                wisdom_score = 0.7
            
            # Fairness assumes equal access
            fairness_score = 0.7
        else:  # consume
            # Consumption can be less cooperative
            conservation_rate = self._get_conservation_rate(shared_resource)
            cooperation_score = max(0.2, conservation_rate - 0.2)
            
            # Moderation is lower for consumption
            if shared_resource.amount < 40:
                moderation_score = 0.3  # Not moderate to consume when resources are low
            else:
                moderation_score = 0.5
            
            # Wisdom depends on resource state
            if self._is_resource_declining(shared_resource):
                wisdom_score = 0.3  # Less wise to consume when resources are declining
            else:
                wisdom_score = 0.5
            
            # Fairness depends on others' actions
            if all(a.action == "conserve" for a in shared_resource.agents if a.id != agent.id):
                fairness_score = 0.2  # Not fair to consume when all others conserve
            else:
                fairness_score = 0.5
        
        # Combine virtue scores
        virtue_score = (
            cooperation_score * 0.3 +
            moderation_score * 0.2 +
            wisdom_score * 0.3 +
            fairness_score * 0.2
        )
        
        # Adjust based on character history (past virtues)
        return 0.7 * virtue_score + 0.3 * sum(self.virtues.values()) / len(self.virtues)
    
    def _update_virtues(self, agent, shared_resource):
        """Update virtue assessments based on history."""
        if not agent.action_history:
            return
        
        # Assess cooperation based on conservation rate
        conserve_count = agent.action_history.count("conserve")
        total_actions = len(agent.action_history)
        self.virtues["cooperation"] = conserve_count / total_actions
        
        # Assess moderation based on balance of actions
        balance = min(conserve_count, total_actions - conserve_count) / (total_actions / 2)
        self.virtues["moderation"] = balance
        
        # Assess wisdom based on appropriate responses to resource state
        correct_responses = 0
        for i, action in enumerate(agent.action_history):
            if i > 0 and hasattr(shared_resource, 'state_log') and i < len(shared_resource.state_log):
                state = shared_resource.state_log[i]
                if state.get('amount', 50) < 30 and action == "conserve":
                    correct_responses += 1
                elif state.get('amount', 50) > 80 and action == "consume":
                    correct_responses += 1
        
        if total_actions > 1:
            self.virtues["wisdom"] = correct_responses / (total_actions - 1)
        
        # Assess fairness
        self.virtues["fairness"] = 0.5  # Default, hard to assess from action history alone
    
    def _is_resource_declining(self, shared_resource) -> bool:
        """Check if the resource is on a declining trend."""
        if hasattr(shared_resource, 'state_log') and len(shared_resource.state_log) > 2:
            recent_states = shared_resource.state_log[-3:]
            amounts = [state.get('amount', 50) for state in recent_states]
            return amounts[0] > amounts[-1]
        return False
    
    def _get_conservation_rate(self, shared_resource) -> float:
        """Get the current conservation rate among all agents."""
        if not shared_resource.agents:
            return 0.5
        
        conserve_count = sum(1 for a in shared_resource.agents if a.action == "conserve")
        return conserve_count / len(shared_resource.agents)
    
    def get_explanation(self, score: float) -> str:
        if score > 0.7:
            return "demonstrates virtuous character traits like cooperation and wisdom"
        elif score > 0.4:
            return "shows moderate virtue in decision-making"
        else:
            return "fails to exhibit virtuous character in this context"
