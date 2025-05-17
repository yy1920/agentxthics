"""
Metrics implementation for AgentXthics research.
These functions calculate various ethical and resource metrics.
"""
import math
from typing import Dict, List, Any, Optional

def calculate_fairness(distribution: Dict[str, float], metric: str = 'gini') -> float:
    """
    Calculate fairness metrics from a resource distribution.
    
    Args:
        distribution: Dictionary mapping agent IDs to resource amounts
        metric: Metric to use ('gini', 'theil', or 'atkinson')
        
    Returns:
        Fairness metric value (0 = perfect equality, 1 = perfect inequality)
    """
    values = list(distribution.values())
    if not values or sum(values) == 0:
        return 0.0
    
    if metric == 'gini':
        return calculate_gini(values)
    elif metric == 'theil':
        return calculate_theil(values)
    elif metric == 'atkinson':
        return calculate_atkinson(values)
    else:
        return calculate_gini(values)  # Default to Gini

def calculate_gini(values: List[float]) -> float:
    """
    Calculate Gini coefficient of inequality.
    
    Args:
        values: List of values (e.g., resource amounts for each agent)
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    # Sort values
    values = sorted(values)
    n = len(values)
    
    # Calculate Gini coefficient
    cumsum = 0
    for i, value in enumerate(values):
        cumsum += (2 * i - n + 1) * value
    
    return cumsum / (n * n * sum(values) / n) if sum(values) > 0 else 0

def calculate_theil(values: List[float]) -> float:
    """
    Calculate Theil index of inequality.
    
    Args:
        values: List of values (e.g., resource amounts for each agent)
        
    Returns:
        Theil index (0 = perfect equality, increases with inequality)
    """
    n = len(values)
    if not n:
        return 0.0
        
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    
    # Calculate Theil index
    theil = 0
    for value in values:
        if value > 0:  # Avoid log(0)
            theil += (value / mean) * math.log(value / mean)
    
    return theil / n

def calculate_atkinson(values: List[float], epsilon: float = 0.5) -> float:
    """
    Calculate Atkinson index of inequality.
    
    Args:
        values: List of values (e.g., resource amounts for each agent)
        epsilon: Inequality aversion parameter (higher = more aversion)
        
    Returns:
        Atkinson index (0 = perfect equality, 1 = perfect inequality)
    """
    n = len(values)
    if not n:
        return 0.0
        
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    
    if epsilon == 1:
        # Special case for epsilon = 1
        product = 1
        for value in values:
            if value > 0:  # Avoid log(0)
                product *= value ** (1/n)
        
        return 1 - (product / mean) if mean > 0 else 0
    else:
        # General case
        sum_term = 0
        for value in values:
            sum_term += (value / mean) ** (1 - epsilon)
        
        return 1 - (sum_term / n) ** (1 / (1 - epsilon))

def calculate_sustainability(amount: float, threshold: float = 30) -> float:
    """
    Calculate sustainability metric for resource amount.
    
    Args:
        amount: Current resource amount
        threshold: Sustainability threshold (resource level considered critical)
        
    Returns:
        Sustainability score (0 = not sustainable, 1 = fully sustainable)
    """
    if amount <= 0:
        return 0.0  # Already collapsed
    elif amount < threshold:
        # Linear scaling between 0 and threshold
        return amount / threshold
    else:
        return 1.0  # Fully sustainable

def calculate_welfare(distribution: Dict[str, float], method: str = 'sum') -> float:
    """
    Calculate welfare metric from resource distribution.
    
    Args:
        distribution: Dictionary mapping agent IDs to resource amounts
        method: Welfare calculation method ('sum', 'min', or 'rawlsian')
        
    Returns:
        Welfare score
    """
    values = list(distribution.values())
    if not values:
        return 0.0
    
    if method == 'sum':
        # Utilitarian welfare: sum of utilities
        return sum(values)
    elif method == 'min':
        # Rawlsian welfare: welfare of worst-off agent
        return min(values)
    elif method == 'rawlsian':
        # Modified Rawlsian: weighted by position (more weight to worse-off)
        values = sorted(values)
        n = len(values)
        weights = [(n - i) / sum(range(1, n + 1)) for i in range(n)]
        return sum(v * w for v, w in zip(values, weights))
    else:
        return sum(values)  # Default to sum

def calculate_ethical_impact(resource, ethical_frameworks: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate impact of ethical reasoning on agent decisions.
    
    Args:
        resource: Resource object after simulation completion
        ethical_frameworks: List of frameworks to analyze (if None, analyze all)
        
    Returns:
        Dictionary with ethical impact metrics
    """
    # Initialize counters
    total_decisions = 0
    changed_decisions = 0
    framework_influences = {}
    framework_scores = {}
    
    # Count decisions changed by ethical reasoning
    for agent in resource.agents:
        if not hasattr(agent, 'ethical_reasoning') or not agent.ethical_reasoning:
            continue
            
        for reasoning in agent.ethical_reasoning:
            total_decisions += 1
            
            # Check if action was changed by ethical reasoning
            if reasoning['proposed_action'] != agent.action_history[reasoning['round']]:
                changed_decisions += 1
                
                # Identify which framework had the most influence
                top_framework, top_data = max(
                    reasoning['evaluations'].items(), 
                    key=lambda x: x[1]['score']
                )
                
                framework_influences[top_framework] = framework_influences.get(top_framework, 0) + 1
            
            # Collect scores for each framework
            for framework, data in reasoning['evaluations'].items():
                if framework not in framework_scores:
                    framework_scores[framework] = []
                    
                framework_scores[framework].append(data['score'])
    
    # Calculate average scores for each framework
    average_scores = {
        framework: sum(scores) / len(scores) if scores else 0
        for framework, scores in framework_scores.items()
    }
    
    # Calculate change rate
    change_rate = changed_decisions / total_decisions if total_decisions > 0 else 0
    
    # Calculate framework influence rates
    influence_rates = {
        framework: count / changed_decisions if changed_decisions > 0 else 0
        for framework, count in framework_influences.items()
    }
    
    return {
        'total_decisions': total_decisions,
        'changed_decisions': changed_decisions,
        'change_rate': change_rate,
        'framework_influences': framework_influences,
        'influence_rates': influence_rates,
        'average_scores': average_scores
    }
