# Electricity Trading Game

This implementation provides a sophisticated multi-agent simulation of electricity trading companies operating in a competitive market environment. The simulation tests specific hypotheses about agent communication, emergent behavior (including deception and collaboration), and safety-relevant decision-making.

## Key Features

- Multiple agents with different personalities and strategies
- Bilateral contract negotiation between agents
- Electricity generation, demand, and storage mechanics
- Market-based auction system for electricity trading
- Communication and trust-building between agents
- Analysis tools to identify and quantify emergent behaviors

## Implementation Architecture

The implementation consists of several components:

1. **ElectricityMarket** - Core market simulation engine that manages the trading environment
2. **ElectricityAgent** - Agent implementation that extends BaseAgent with electricity-specific logic
3. **ElectricityTradingGame** - Main controller that sets up and runs the simulation
4. **Analysis Tools** - Components that analyze the results to test specific hypotheses

## Running the Simulation

You can run the simulation using the provided runner script:

```bash
python run_electricity_trading.py
```

### Command Line Options

- `--config FILENAME` - Specify a custom configuration file (default: config_electricity_trading.json)
- `--analyze-only` - Only analyze results without running a new simulation
- `--run-dir DIRECTORY` - Specify a run directory to analyze (used with --analyze-only)

Example:
```bash
# Run a new simulation with default settings
python run_electricity_trading.py

# Run with a custom configuration
python run_electricity_trading.py --config my_config.json

# Analyze a previous run
python run_electricity_trading.py --analyze-only --run-dir logs/electricity_trading/run_20250515-123456
```

## Configuration

The simulation is configured through a JSON file (`config_electricity_trading.json`). Key parameters include:

- Market settings (initial price, volatility, number of rounds)
- Agent profiles (generation capacity, storage capacity, personalities)
- Market conditions (scarcity, demand fluctuations)
- Communication settings

Example configuration:
```json
{
  "market": {
    "initial_price": 40,
    "price_volatility": 0.2,
    "num_rounds": 20
  },
  "agents": [
    {
      "id": "A",
      "personality": "cooperative",
      "profit_bias": 0.3,
      "generation_capacity": 100,
      "storage_capacity": 50,
      "demand_profile": "steady"
    },
    {
      "id": "B",
      "personality": "competitive",
      "profit_bias": 0.7,
      "generation_capacity": 120,
      "storage_capacity": 60,
      "demand_profile": "variable"
    }
  ]
}
```

## Hypotheses Tested

This simulation tests the following hypotheses as specified in the original proposal:

1. **Communication Impact**: Text communication between agents impacts their decision making as opposed to just historic decisions and other data
2. **Relationship Formation**: Agents form relationships with one another based on historic behaviour
3. **Trust and Collaboration**: Agents aim for collaboration based on trust
4. **Deception**: Agents will use deception to maximize profits
5. **Profit vs. Stability**: Agents will prioritize profits over market stability
6. **Emergent Behavior**: Agents will display emergent behaviour when faced with unpredictable market behavior
7. **Decision Quality**: Agents will always make the most logical decisions based on their circumstances
8. **Numerical Accuracy**: Agents will not mistake numerical amounts

## Analysis and Results

After running the simulation, detailed analysis is saved to the output directory:

- `message_log.json` - All communications between agents
- `decision_log.json` - All decisions made by agents
- `contract_log.json` - All contracts between agents
- `state_log.json` - Market state at each turn
- `summary.json` - Aggregated metrics and summary
- `analysis.json` - Detailed analysis of hypotheses
- `analysis_report.txt` - Human-readable report

## Implementation Notes

- Agents use LLM-based decision making that can be configured to use OpenAI, Gemini, or a mock implementation
- Market dynamics include scarcity conditions, price volatility, and unpredictable demand
- The contract system allows negotiation, acceptance, rejection, and counter-offers
- The auction system provides a secondary market for excess electricity
- Analysis tools track trust formation, deception, emergent behavior, and decision quality
