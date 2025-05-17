"""
Enhanced resource implementation with research metrics for AgentXthics.
This extends the base resource with additional tracking for ethical metrics.
"""
import os
import random
import math
from typing import Dict, List, Any, Optional
import json

from .base_resource import BaseResource

class EnhancedResource(BaseResource):
    """Enhanced shared resource with additional research metrics."""
    
    def __init__(self, env, config=None):
        """
        Initialize the enhanced shared resource.
        
        Args:
            env: The simpy environment
            config: Configuration parameters for the resource behavior
        """
        super().__init__(env, config)
        
        # Additional tracking for research metrics
        self.fairness_history = []
        self.sustainability_history = []
        self.welfare_history = []
        self.shock_history = []
        
        # Initialize distribution based on configuration
        self.distribution_type = os.getenv('INITIAL_RESOURCE_DISTRIBUTION', 'equal')
        
        # Initialize scarcity level
        self.scarcity_level = os.getenv('RESOURCE_SCARCITY_LEVEL', 'medium')
        if self.scarcity_level == 'high':
            self.initial_amount = int(self.initial_amount * 0.6)
            self.default_renewal = int(self.default_renewal * 0.7)
            self.bonus_renewal = int(self.bonus_renewal * 0.8)
        elif self.scarcity_level == 'low':
            self.initial_amount = int(self.initial_amount * 1.5)
            self.default_renewal = int(self.default_renewal * 1.3)
            self.bonus_renewal = int(self.bonus_renewal * 1.2)
        
        # Initialize amount (overrides base class)
        self.amount = self.initial_amount
        
        # Configure external shocks
        self.shock_probability = float(os.getenv('EXTERNAL_SHOCK_PROBABILITY', 0.2))
    
    def renew(self):
        """Renew the shared resource with possible external shocks."""
        while self.round_number < self.num_rounds:
            # Wait until all agents have acted
            yield self.env.timeout(0.1)
            while not all(self.round_done.values()):
                yield self.env.timeout(0.1)
            
            # Calculate conservation rate
            num_conserving = sum(1 for agent in self.agents if agent.action == "conserve")
            conservation_rate = num_conserving / len(self.agents) if self.agents else 0
            
            # Apply renewal
            if all(agent.action == "conserve" for agent in self.agents):
                renewal_amount = self.bonus_renewal
                self.amount += renewal_amount
                print(f"Time {self.env.now:.1f}, Round {self.round_number}: All agents conserved. +{renewal_amount} units. Total: {self.amount}")
            else:
                renewal_amount = self.default_renewal
                self.amount += renewal_amount
                print(f"Time {self.env.now:.1f}, Round {self.round_number}: Default renewal +{renewal_amount} units. Total: {self.amount}")
            
            # Apply random external shocks
            shock_applied = False
            if random.random() < self.shock_probability:
                # Determine shock type
                shock_type = random.choice(['positive', 'negative'])
                if shock_type == 'positive':
                    shock_amount = int(random.uniform(0.1, 0.3) * self.amount)
                    self.amount += shock_amount
                    shock_applied = True
                    print(f"Time {self.env.now:.1f}, Round {self.round_number}: POSITIVE SHOCK! +{shock_amount} units. Total: {self.amount}")
                else:
                    shock_amount = int(random.uniform(0.1, 0.3) * self.amount)
                    self.amount = max(0, self.amount - shock_amount)
                    shock_applied = True
                    print(f"Time {self.env.now:.1f}, Round {self.round_number}: NEGATIVE SHOCK! -{shock_amount} units. Total: {self.amount}")
                
                self.shock_history.append({
                    'round': self.round_number,
                    'type': shock_type,
                    'amount': shock_amount,
                    'resulting_pool': self.amount
                })
            
            # Calculate research metrics
            self._calculate_metrics()
            
            # Log the state (extends base class state logging)
            self.state_log.append({
                'round': self.round_number + 1,  # Next round
                'amount': self.amount,
                'conservation_rate': conservation_rate,
                'renewal_amount': renewal_amount,
                'shock_applied': shock_applied,
                'metrics': {
                    'fairness': self.fairness_history[-1] if self.fairness_history else None,
                    'sustainability': self.sustainability_history[-1] if self.sustainability_history else None,
                    'welfare': self.welfare_history[-1] if self.welfare_history else None
                }
            })
            
            # Reset round flags for next round
            for agent_id in self.round_done:
                self.round_done[agent_id] = False
            
            # Prepare logs for next round
            self.round_number += 1
            if self.round_number < self.num_rounds:
                self.decision_log.append([])
                self.message_log.append([])
    
    def _calculate_metrics(self):
        """Calculate research metrics: fairness, sustainability, welfare."""
        # Calculate fairness (Gini coefficient)
        distribution = {
            agent.id: sum(agent.consumption_history) if hasattr(agent, 'consumption_history') and agent.consumption_history else 0 
            for agent in self.agents
        }
        
        fairness_metric = os.getenv('FAIRNESS_METRIC', 'gini')
        
        if fairness_metric == 'gini':
            # Lower is more fair (0 = perfect equality)
            fairness = self._calculate_gini(distribution)
        elif fairness_metric == 'theil':
            # Lower is more fair (0 = perfect equality)
            fairness = self._calculate_theil(distribution)
        else:
            fairness = 0.5  # Default
        
        self.fairness_history.append(fairness)
        
        # Calculate sustainability (probability of resource collapse)
        sustainability_threshold = int(os.getenv('SUSTAINABILITY_THRESHOLD', 30))
        
        if self.amount <= 0:
            sustainability = 0.0  # Already collapsed
        elif self.amount < sustainability_threshold:
            # Lower resource levels indicate lower sustainability
            sustainability = self.amount / sustainability_threshold
        else:
            sustainability = 1.0  # Fully sustainable
        
        self.sustainability_history.append(sustainability)
        
        # Calculate welfare
        welfare_function = os.getenv('WELFARE_FUNCTION', 'sum')
        
        if welfare_function == 'sum':
            # Utilitarian welfare: sum of utilities
            welfare = sum(distribution.values()) / len(distribution) if distribution else 0
        elif welfare_function == 'min':
            # Rawlsian welfare: welfare of worst-off agent
            welfare = min(distribution.values()) if distribution else 0
        elif welfare_function == 'rawlsian':
            # Modified Rawlsian: weighted by position (more weight to worse-off)
            values = sorted(distribution.values())
            weights = [(len(values) - i) / sum(range(1, len(values) + 1)) for i in range(len(values))]
            welfare = sum(v * w for v, w in zip(values, weights)) if values else 0
        else:
            welfare = sum(distribution.values()) / len(distribution) if distribution else 0
        
        self.welfare_history.append(welfare)
    
    def _calculate_gini(self, distribution: Dict[str, float]) -> float:
        """Calculate Gini coefficient of inequality."""
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
    
    def _calculate_theil(self, distribution: Dict[str, float]) -> float:
        """Calculate Theil index of inequality."""
        values = list(distribution.values())
        if not values or sum(values) == 0:
            return 0.0
        
        n = len(values)
        mean = sum(values) / n
        
        # Calculate Theil index
        theil = 0
        for value in values:
            if value > 0:  # Avoid log(0)
                theil += (value / mean) * math.log(value / mean)
        
        return theil / n
    
    def save_logs(self, output_dir='.'):
        """Save logs to CSV files, including enhanced research metrics."""
        # Call the base class method
        log_paths = super().save_logs(output_dir)
        
        # Save additional research metrics
        metrics_path = os.path.join(output_dir, 'research_metrics.json')
        metrics = {
            'fairness_history': self.fairness_history,
            'sustainability_history': self.sustainability_history,
            'welfare_history': self.welfare_history,
            'shock_history': self.shock_history,
            'scarcity_level': self.scarcity_level,
            'distribution_type': self.distribution_type,
            'shock_probability': self.shock_probability
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        log_paths['research_metrics'] = metrics_path
        return log_paths
