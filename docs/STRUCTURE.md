# AgentXthics Project Structure

This document outlines the organization of the AgentXthics project, explaining the purpose of each directory and the files they contain.

## Top-Level Organization

```
agentxthics_organized/
├── agentxthics/               # Main package source code
├── docs/                      # Documentation files
├── examples/                  # Example configurations and usage
├── logs/                      # Log files from simulations
├── results/                   # Simulation results
│   ├── electricity_game/      # Results from electricity game
│   └── other_scenarios/       # Results from other scenarios
├── scripts/                   # Utility scripts
└── tests/                     # Test suites and utilities
```

## Main Package Structure

The `agentxthics` package is organized into several modules:

```
agentxthics/
├── agents/                    # Agent implementations
├── frameworks/                # Ethical frameworks
├── llm/                       # Language model integrations
├── research/                  # Research methods and tools
├── resources/                 # Resource models and implementations
├── scenarios/                 # Predefined simulation scenarios
├── utils/                     # Utility functions and helpers
└── visualization/             # Visualization and dashboard tools
```

### Agents

Contains agent implementations for simulations:
- `base_agent.py`: Base agent class with core functionality
- `enhanced_agent.py`: Enhanced agent with ethical reasoning capabilities

### Frameworks

Contains ethical frameworks for agent decision making:
- `base.py`: Base ethical framework class
- `utilitarian.py`: Utilitarian ethics framework
- `deontological.py`: Deontological ethics framework
- `virtue.py`: Virtue ethics framework
- `care.py`: Care ethics framework
- `justice.py`: Justice ethics framework

### LLM (Language Model)

Contains integrations with various language models:
- `base_llm.py`: Base language model interface
- `mock_llm.py`: Mock implementation for testing
- `gemini_llm.py`: Integration with Google's Gemini AI

### Research

Contains research tools and methods:
- `scenarios.py`: Simulation scenario definitions
- `analysis.py`: Analysis of research results
- `metrics.py`: Metrics calculations for simulations

### Resources

Contains shared resource implementations:
- `base_resource.py`: Base shared resource implementation
- `enhanced_resource.py`: Enhanced resource with additional features

### Scenarios

Contains predefined simulation scenarios:
- Various scenario implementations for running simulations

### Utils

Contains utility functions and tools:
- `analyze_decisions.py`: Analysis of agent decisions
- `analyze_messages.py`: Analysis of agent messages
- `analyze_logs.py`: Analysis of simulation logs

### Visualization

Contains visualization and dashboard tools:
- `dashboard.py`: Dashboard for visualizing simulation results

## Entry Points

The following files are key entry points to the project:

- `setup.py`: Package installation configuration
- `requirements.txt`: Dependencies list
- `simple_run.py`: Simplified entry point for running simulations
- `run_research.py`: Main entry point for running research simulations

## Example Usage

To run a basic simulation:

```bash
python simple_run.py
```

To run a specific research scenario:

```bash
python run_research.py --scenario asymmetric
```

## Configuration

The project can be configured via environment variables or a `.env` file. See `.env.sample` for an example configuration.

## Logs

Simulation logs are stored in the `logs` directory, organized by run type:

- `current_run/`: Logs from the most recent simulation
- `debug_run/`: Logs from debug runs
