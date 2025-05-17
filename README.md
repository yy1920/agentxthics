# AgentXthics

A multi-agent simulation framework for researching ethical decision-making in resource management scenarios.

## Overview

AgentXthics allows researchers to model and simulate how agents with different ethical frameworks make decisions about shared resources. The framework includes:

- Multiple agent types with customizable behavior
- Various ethical frameworks for evaluating decisions
- Resource management simulations with environmental shocks
- Detailed logging and visualization tools
- Integration with LLMs for agent decision-making

## Ethical Frameworks Research

The AgentXthics project's primary research focus is exploring how explicit ethical frameworks influence agent behavior in competitive resource management environments.
See `ethical_frameworks_simulations` for more details.

### Research Objective 

This research investigates what outcomes emerge when agents explicitly adhere to different established ethical frameworks within competitive environments. The goal is to compare how different ethical approaches influence individual behavior and system-level outcomes including resource sustainability, fairness, and system stability.

### Ethical Frameworks Implemented

The simulation includes agents that follow several ethical frameworks:

- **Utilitarianism**: Agents attempt to maximize the total well-being or utility of the entire system
- **Deontological Ethics**: Agents follow moral duties and obligations regardless of outcome
- **Virtue Ethics**: Agents act according to virtuous character traits
- **Care Ethics**: Agents prioritize relationships and caring for others
- **Justice**: Agents focus on fairness and equitable distribution of resources

## Installation

There are two ways to set up the AgentXthics environment:

### Option 1: Quick Installation (using requirements.txt)

1. Clone this repository
2. Create and activate a virtual environment (see below)
3. Install dependencies: `pip install -r requirements.txt`

### Option 2: Development Installation (using setup.py)

This method installs the AgentXthics package in development mode and creates command-line tools:

1. Clone this repository
2. Create and activate a virtual environment (see below)
3. Install in development mode with all features:
   ```bash
   pip install -e ".[dev,llm]"
   ```

### Environment Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration Files

Two configuration files are required to run the simulations:

1. **Environment Variables (.env)**:
   - Copy the `.env.sample` file to create a `.env` file
   - Add your actual API keys if you want to use OpenAI or Gemini

   Example `.env` file:
   ```
   # LLM API Keys
   GEMINI_API_KEY=your-gemini-api-key-here
   OPENAI_API_KEY=your-openai-api-key-here
   ```

2. **Simulation Configuration (config_ethical_frameworks.json)**:
   - This configuration file sets up the ethical frameworks experiment
   - It creates agents using different ethical frameworks and configures the resource sharing scenario

## Running the Ethical Frameworks Experiment

The experiment can be run with the following command:

```bash
python run_ethical_frameworks.py --llm gpt --rounds 10 --initial-amount 80
```

### Command Line Parameters

- `--llm`: Specifies the LLM provider to use (options: `gpt`, `gemini`, `mock`)
- `--rounds`: Sets the number of simulation rounds to run
- `--initial-amount`: Sets the initial resource pool amount
- `--output`: (Optional) Specifies a custom output directory

### Example Run Commands

```bash
# Run with default settings
python run_ethical_frameworks.py

# Run with OpenAI GPT and 5 rounds
python run_ethical_frameworks.py --llm gpt --rounds 5

# Run with mock LLM (no API calls) and a large resource pool
python run_ethical_frameworks.py --llm mock --initial-amount 100

# Run with a specific output directory
python run_ethical_frameworks.py --output custom_results
```

## Simulation Output and Analysis

All experiment results and analysis files are stored in timestamped directories under:
```
game_logs/ethical_frameworks/run_YYYYMMDD_HHMMSS/

EXAMPLE: /Users/debjyotiray/projects/agentx/agentxthics_organized/game_logs/ethical_frameworks
```

### Generated Log Files

The simulation generates several log files:

1. **Raw Data**:
   - `decision_log.json`: JSON record of all agent decisions and explanations
   - `message_log.json`: JSON record of all inter-agent communications
   - `state_log.json`: Resource state history by round
   - `timeout_log.json`: Records of any timeout events
   - `public_log.txt`: Plain text log of all public events

2. **Analyzed Data**:
   - `agent_decisions.txt`: Human-readable summary of all agent decisions, organized by round and agent
   - `agent_conversations.txt`: Human-readable log of all agent communications
   - `analysis_summary.txt`: Overall analysis of agent behavior patterns
   - `decision_analysis.png`: Visualization of decision patterns
   - `message_analysis.png`: Visualization of communication patterns

### Understanding the Analysis

#### Decision Log Analysis

The `agent_decisions.txt` file provides human-readable insights into agent decisions:
- For each round, you'll see each agent's decision (conserve or consume)
- The justification from the agent's ethical framework
- The reasoning process that led to the decision

For example:
```
=== ROUND 2 ===
U1: conserve - Given the current low resource pool of 20 units and my cooperation bias of 0.6, it's important to balance self-interest with the well-being of the collective... [Utilitarian reasoning: maximizes total welfare for all agents]
```

The decision analysis helps researchers understand:
- How different ethical frameworks respond to resource constraints
- Whether frameworks tend to be more or less cooperative
- How decisions change over time and in response to other agents

#### Message Analysis

The `agent_conversations.txt` file shows all agent communications:
- Messages are organized by round
- Each message shows sender, recipient, and content
- Messages reveal negotiation strategies and persuasion tactics

For example:
```
=== ROUND 0 ===
D1 -> J1: Let's both use 15 units each to secure immediate gains while ensuring resources remain for future rounds.
```

The message analysis reveals:
- How agents with different frameworks communicate
- Persuasion strategies employed by different frameworks
- Trust building and negotiation patterns

#### Visual Analysis

The generated visualizations provide additional insights:
- **Decision Analysis (decision_analysis.png)**: Shows patterns in conservation/consumption behavior by agent and round
- **Message Analysis (message_analysis.png)**: Displays communication network, message volume, and content analysis

### Dashboard for Results Exploration

For a more interactive exploration of results, the system generates dashboard files:

1. **Interactive Dashboard**:
   ```bash
   python game_logs/ethical_frameworks/run_YYYYMMDD_HHMMSS/launch_dashboard.py
   ```
   
2. **Static Dashboard** (no server required):
   Simply open the following file in any web browser:
   ```
   game_logs/ethical_frameworks/run_YYYYMMDD_HHMMSS/dashboard_static.html
   ```

## LLM Integration

AgentXthics supports integration with various LLMs for agent decision making. The framework currently supports:

- **OpenAI GPT**: High-quality reasoning with complex ethical frameworks
- **Google Gemini**: Alternative LLM with different reasoning patterns
- **Mock LLM**: Rule-based decisions for testing without API costs

The LLM integration is used in two key ways:
1. **Decision Making**: Agents use the LLM to reason through ethical dilemmas and make conservation/consumption decisions
2. **Communication**: Agents use the LLM to generate messages to other agents to persuade them to cooperate

## Project Structure

This project has been organized into a modular structure:

```
agentxthics_organized/
├── agentxthics/           # Main package source
│   ├── agents/            # Agent implementations
│   ├── frameworks/        # Ethical frameworks
│   ├── llm/               # Language model integrations
│   ├── research/          # Research methods and tools
│   ├── resources/         # Resource models
│   ├── utils/             # Utility functions
│   └── visualization/     # Visualization tools
├── game_logs/             # Simulation logs and results
│   └── ethical_frameworks/# Ethical frameworks experiment results
├── docs/                  # Documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
