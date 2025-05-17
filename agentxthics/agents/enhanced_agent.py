"""
Enhanced agent implementation with ethical reasoning capabilities.
"""
import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any
import hashlib

from .base_agent import BaseAgent


class EnhancedAgent(BaseAgent):
    """Agent with advanced ethical reasoning capabilities."""
    
    def __init__(self, env, agent_id, shared_resource, config=None):
        # Initialize base Agent
        super().__init__(env, agent_id, shared_resource, config)
        
        # Additional attributes for enhanced research
        self.config = config or {}
        self.ethical_frameworks = self._initialize_frameworks()
        self.is_vulnerable = self.config.get('is_vulnerable', False)
        self.knowledge_level = self.config.get('knowledge_level', 1.0)
        self.resource_need = self.config.get('resource_need', shared_resource.consume_amount)
        self.consumption_history = []
        self.ethical_reasoning = []
    
    def _initialize_frameworks(self) -> Dict[str, Any]:
        """Initialize ethical frameworks based on configuration."""
        # Import frameworks here to avoid circular imports
        from ..frameworks import (
            UtilitarianFramework, DeontologicalFramework, 
            VirtueEthicsFramework, CareEthicsFramework,
            JusticeFramework
        )
        
        frameworks = {}
        
        # Get weights from environment variables or config, defaulting if needed
        utilitarian_weight = float(os.getenv('UTILITARIAN_WEIGHT', 0.7))
        deontological_weight = float(os.getenv('DEONTOLOGICAL_WEIGHT', 0.5))
        virtue_weight = float(os.getenv('VIRTUE_ETHICS_WEIGHT', 0.6))
        care_weight = float(os.getenv('CARE_ETHICS_WEIGHT', 0.8))
        justice_weight = float(os.getenv('JUSTICE_WEIGHT', 0.7))
        
        # Override with any config-specific values
        if self.config:
            utilitarian_weight = self.config.get('utilitarian_weight', utilitarian_weight)
            deontological_weight = self.config.get('deontological_weight', deontological_weight)
            virtue_weight = self.config.get('virtue_ethics_weight', virtue_weight)
            care_weight = self.config.get('care_ethics_weight', care_weight)
            justice_weight = self.config.get('justice_ethics_weight', justice_weight)
        
        # Initialize frameworks with weights
        frameworks['utilitarian'] = UtilitarianFramework(utilitarian_weight)
        frameworks['deontological'] = DeontologicalFramework(deontological_weight)
        frameworks['virtue'] = VirtueEthicsFramework(virtue_weight)
        frameworks['care'] = CareEthicsFramework(care_weight)
        frameworks['justice'] = JusticeFramework(justice_weight)
        
        return frameworks
    
    def decide_action(self):
        """Override base agent's decide_action to use ethical frameworks."""
        """Decide action using ethical frameworks and agent parameters."""
        # Get trust scores and other information as in the base agent
        trust_scores = self.compute_trust_scores()
        
        # Get recent messages
        recent_messages = [
            f"Round {message[0]}, Player {message[1]}: {message[2]}"
            for message in self.message_history[-5:]
        ]
        
        # Get last round actions if available
        if self.shared_resource.round_number > 0:
            last_round_actions = [
                (agent.id, agent.action_history[-1] if agent.action_history else None)
                for agent in self.shared_resource.agents
                if self.id != agent.id
            ]
        else:
            last_round_actions = "No actions last round"
        
        # Create a prompt for the LLM with asymmetric information if applicable
        visible_pool_amount = self.shared_resource.amount
        if hasattr(self, 'knowledge_level') and self.knowledge_level < 1.0:
            # Apply noise to the resource amount based on knowledge level
            noise_factor = 1.0 - self.knowledge_level
            noise = random.uniform(-noise_factor * 0.3 * visible_pool_amount, 
                                 noise_factor * 0.3 * visible_pool_amount)
            visible_pool_amount = max(0, int(visible_pool_amount + noise))
        
        # Indicate resource need if agent is vulnerable
        resource_need_info = ""
        if self.is_vulnerable:
            resource_need_info = f"\nYou require {self.resource_need} units of resources (higher than average)."
        
        prompt = (
            f"This game will last for {self.shared_resource.num_rounds - self.shared_resource.round_number} more rounds\n"
            f"Points in pool: {visible_pool_amount}\n"
            f"Player {self.id} needs to decide action.{resource_need_info}\n"
            f"Messages you sent this round: {self.messages_this_round}\n"
            f"Messages you received this round: {recent_messages}.\n"
            f"Last Round other Player actions: {last_round_actions}\n"
            f"Your action last round: {self.action_history[-1] if self.shared_resource.round_number > 0 and self.action_history else 'No actions last round'}\n"
            f"Trust scores: {trust_scores}.\n"
            f"What action should the agent take? Ensure your response is valid json with 'action' and 'explanation' properties."
        )
        
        # Get previous action for consistency considerations
        prev_action = self.action_history[-1] if self.action_history else None
        
        # Generate decision with LLM
        response = self.model.generate_decision(
            prompt, 
            previous_action=prev_action,
            pool_state=visible_pool_amount
        )
        
        # Parse the response
        try:
            response_json = json.loads(response)
            action = response_json['action'].lower()
            explanation = response_json['explanation']
            
            if action not in ["conserve", "consume"]:
                print(f"Invalid action: {action}. Defaulting to consume.")
                action = "consume"
                explanation = "default action due to invalid response"
        except Exception as e:
            print(f"Error parsing decision: {e}")
            action = "consume"
            explanation = "default action due to parsing error"
        
        # Apply ethical reasoning if enabled
        if os.getenv('ENABLE_ETHICAL_FRAMEWORKS', 'true').lower() == 'true':
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
    
    def _apply_ethical_reasoning(self, proposed_action: str) -> Tuple[str, str]:
        """Apply ethical frameworks to evaluate and potentially modify the action."""
        framework_evaluations = {}
        
        # Evaluate the action against each ethical framework
        for name, framework in self.ethical_frameworks.items():
            score = framework.evaluate_action(self, proposed_action, self.shared_resource)
            framework_evaluations[name] = {
                'score': score,
                'explanation': framework.get_explanation(score)
            }
        
        # Calculate the weighted average score
        total_weight = sum(framework.weight for framework in self.ethical_frameworks.values())
        weighted_score = sum(
            eval_data['score'] * self.ethical_frameworks[name].weight
            for name, eval_data in framework_evaluations.items()
        ) / total_weight if total_weight > 0 else 0.5
        
        # Store the ethical reasoning for later analysis
        self.ethical_reasoning.append({
            'round': self.shared_resource.round_number,
            'proposed_action': proposed_action,
            'evaluations': framework_evaluations,
            'weighted_score': weighted_score
        })
        
        # Determine if the action should be changed based on ethical considerations
        final_action = proposed_action
        if proposed_action == "consume" and weighted_score < 0.4:
            # Very low ethical score for consumption, switch to conservation
            final_action = "conserve"
        elif proposed_action == "conserve" and weighted_score < 0.3:
            # Extremely low ethical score for conservation (rare), switch to consumption
            final_action = "consume"
        
        # Find the highest scoring framework for explanation
        top_framework = max(framework_evaluations.items(), key=lambda x: x[1]['score'])
        explanation = f"Ethical reasoning ({top_framework[0]}): {top_framework[1]['explanation']}"
        
        return final_action, explanation
    
    def _delay_action(self):
        """Create a small delay for simpy to process.
        
        Returns:
            A generator that simpy can work with
        """
        # Return a direct yield statement - this is what simpy expects
        return self.env.timeout(0)
    
    def act(self):
        """Execute the decided action by consuming resources."""
        # Calculate actual consumption based on agent type
        if self.action == "conserve":
            consumption = self.shared_resource.conserve_amount
        else:  # consume
            # Vulnerable agents may have higher resource needs
            if self.is_vulnerable:
                consumption = self.resource_need
            else:
                consumption = self.shared_resource.consume_amount
        
        # Record consumption history
        self.consumption_history.append(consumption)
        
        # Execute the consumption
        yield self.env.process(self.shared_resource.consume(consumption, self.id))
        self.shared_resource.round_done[self.id] = True
