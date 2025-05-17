"""
Deontological ethical framework for agent decision making.
This framework evaluates actions based on moral rules and duties.
"""
import re
from typing import Dict, List, Any, Tuple, Optional

from .base import EthicalFramework


class DeontologicalFramework(EthicalFramework):
    """Deontological ethics: Follow moral rules and duties."""
    
    def __init__(self, weight: float = 0.5):
        super().__init__("Deontological", weight)
    
    def evaluate_action(self, agent, action: str, shared_resource) -> float:
        # Rules:
        # 1. Do not deplete common resources
        # 2. Keep promises (past communications)
        # 3. Act consistently
        
        # Rule 1: Don't deplete common resources
        if shared_resource.amount < 30 and action == "consume":
            resource_score = 0.2
        else:
            resource_score = 0.7
        
        # Rule 2: Keep promises from past communications
        promised_action = self._extract_promised_action(agent)
        promise_score = 1.0 if promised_action is None or promised_action == action else 0.2
        
        # Rule 3: Consistency with past actions
        if len(agent.action_history) > 0:
            last_action = agent.action_history[-1]
            consistency_score = 0.7 if action == last_action else 0.4
        else:
            consistency_score = 0.5
        
        # Weight the rules (can be adjusted)
        return 0.4 * resource_score + 0.4 * promise_score + 0.2 * consistency_score
    
    def _extract_promised_action(self, agent) -> Optional[str]:
        """Extract any promised action from recent messages."""
        # Search the last 5 messages for promises
        recent_messages = agent.messages_this_round or ""
        
        conserve_patterns = [
            r"I (?:will|plan to|intend to|am going to) conserve",
            r"I'll conserve",
            r"let's (?:all)? conserve"
        ]
        
        consume_patterns = [
            r"I (?:will|plan to|intend to|am going to) consume",
            r"I'll consume",
            r"I need to consume"
        ]
        
        for pattern in conserve_patterns:
            if re.search(pattern, recent_messages, re.IGNORECASE):
                return "conserve"
                
        for pattern in consume_patterns:
            if re.search(pattern, recent_messages, re.IGNORECASE):
                return "consume"
        
        return None
    
    def get_explanation(self, score: float) -> str:
        if score > 0.7:
            return "follows moral duties and commitments"
        elif score > 0.4:
            return "partially adheres to ethical obligations"
        else:
            return "violates important moral principles"
