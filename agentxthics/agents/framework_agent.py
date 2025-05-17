"""
Framework-driven agent that explicitly follows a specific ethical framework.
"""
import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any
import hashlib

from .enhanced_agent import EnhancedAgent
from ..frameworks.base import EthicalFramework


class FrameworkAgent(EnhancedAgent):
    """
    Agent that explicitly follows a specific ethical framework.
    
    Unlike the EnhancedAgent which uses a weighted approach across multiple frameworks,
    this agent strictly adheres to a single ethical framework for decision making.
    """
    
    def __init__(self, env, agent_id, shared_resource, config=None):
        # Initialize base enhanced agent
        super().__init__(env, agent_id, shared_resource, config)
        
        # Override ethical frameworks to use only the designated framework
        self.primary_framework = self._get_primary_framework()
        self.framework_adherence = self.config.get('framework_adherence', 0.9)
        
        # Store decisions that were overridden by ethical considerations
        self.overridden_decisions = []
    
    def _get_primary_framework(self) -> EthicalFramework:
        """Get the primary ethical framework this agent follows."""
        # Get the framework type from config
        framework_type = self.config.get('framework_type', 'utilitarian').lower()
        
        if framework_type not in self.ethical_frameworks:
            print(f"Warning: Framework type '{framework_type}' not found. Defaulting to utilitarian.")
            framework_type = 'utilitarian'
        
        # Set the weight of the primary framework to 1.0 (full adherence)
        # and other frameworks to 0 to ensure this is the only one used
        primary_framework = self.ethical_frameworks[framework_type]
        primary_framework.weight = 1.0
        
        # Log which framework this agent is using
        print(f"Agent {self.id} initialized with {framework_type} framework")
        
        return primary_framework
    
    def _apply_ethical_reasoning(self, proposed_action: str) -> Tuple[str, str]:
        """
        Apply strict ethical reasoning based on a single framework.
        
        This method overrides the weighted approach in EnhancedAgent to strictly
        follow the designated ethical framework.
        """
        # Evaluate the action against the primary framework
        score = self.primary_framework.evaluate_action(self, proposed_action, self.shared_resource)
        
        # Store the ethical reasoning for later analysis
        self.ethical_reasoning.append({
            'round': self.shared_resource.round_number,
            'proposed_action': proposed_action,
            'framework': self.primary_framework.name,
            'score': score,
            'framework_adherence': self.framework_adherence
        })
        
        # Determine if the action should be changed based on ethical considerations
        # The framework_adherence parameter controls how strictly the agent follows the framework
        final_action = proposed_action
        framework_explanation = self.primary_framework.get_explanation(score)
        
        # If the ethical score is low and we're following the framework strictly,
        # potentially override the proposed action
        threshold = 1.0 - self.framework_adherence
        
        if score < threshold:
            # Low ethical score, consider changing the action
            if proposed_action == "consume":
                final_action = "conserve"
            else:
                final_action = "consume"
            
            # Record that this decision was overridden
            self.overridden_decisions.append({
                'round': self.shared_resource.round_number,
                'original_action': proposed_action,
                'new_action': final_action,
                'score': score
            })
        
        # Format the explanation
        if final_action != proposed_action:
            explanation = f"{self.primary_framework.name} ethical override: {framework_explanation}"
        else:
            explanation = f"{self.primary_framework.name} reasoning: {framework_explanation}"
        
        return final_action, explanation
    
    def decide_action(self):
        """Decide action explicitly following ethical framework."""
        # Get visible pool amount with potential noise based on knowledge level
        visible_pool_amount = self.shared_resource.amount
        if self.knowledge_level < 1.0:
            # Apply noise to the resource amount based on knowledge level
            noise_factor = 1.0 - self.knowledge_level
            noise = random.uniform(-noise_factor * 0.3 * visible_pool_amount, 
                                noise_factor * 0.3 * visible_pool_amount)
            visible_pool_amount = max(0, int(visible_pool_amount + noise))
        
        # Indicate resource need if agent is vulnerable
        resource_need_info = ""
        if self.is_vulnerable:
            resource_need_info = f"\nYou require {self.resource_need} units of resources (higher than average)."
        
        # Get previous action for consistency considerations
        prev_action = self.action_history[-1] if self.action_history else None
        
        # Create a prompt that emphasizes the ethical framework
        framework_prompt = (
            f"You are an agent following {self.primary_framework.name} ethics.\n"
            f"Current resource pool: {visible_pool_amount} units.\n"
            f"Round {self.shared_resource.round_number + 1} of {self.shared_resource.num_rounds}.\n"
            f"Your previous action: {prev_action if prev_action else 'None'}.\n"
            f"As a {self.primary_framework.name} agent, your goal is to make decisions that "
            f"align with this ethical framework.{resource_need_info}\n\n"
            f"According to {self.primary_framework.name} ethics, should you 'conserve' "
            f"or 'consume' resources this round? Explain your reasoning."
        )
        
        # Generate decision with LLM
        response = self.model.generate_decision(
            framework_prompt, 
            previous_action=prev_action,
            pool_state=visible_pool_amount
        )
        
        # Parse the response
        try:
            response_json = json.loads(response)
            action = response_json['action'].lower()
            explanation = response_json['explanation']
            
            if action not in ["conserve", "consume"]:
                print(f"Agent {self.id}: Invalid action '{action}' from LLM. Defaulting to conserve.")
                action = "conserve"
                explanation = "default action due to invalid response"
                
            print(f"Agent {self.id}: LLM decided to {action} based on {self.primary_framework.name} framework")
        except Exception as e:
            print(f"Error parsing decision for agent {self.id}: {e}")
            action = "conserve"
            explanation = f"default action due to parsing error: {str(e)}"
        
        # Apply strict ethical reasoning
        action, ethical_explanation = self._apply_ethical_reasoning(action)
        explanation = f"{explanation} [{ethical_explanation}]"
        
        # Record the decision
        self.action = action
        self.action_history.append(action)
        self.cooperation_history.append(action == "conserve")
        self.update_checksums()
        
        # Log the decision
        self.shared_resource.log_decision(
            self.shared_resource.round_number,
            self.id,
            action,
            explanation
        )
        
        # Just return a direct timeout for simpy
        # Note: BaseAgent.run() will wrap this in a process
        yield self.env.timeout(0)
