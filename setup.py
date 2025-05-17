from setuptools import setup, find_packages

setup(
    name="agentxthics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "simpy>=4.0.1",
        "python-dotenv>=0.19.0",
        
        # Analysis and visualization
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "seaborn>=0.11.0",
        
        # Web dashboard
        "flask>=2.0.0",
        
        # LLM integration - Optional
        "openai>=1.0.0;python_version>='3.8'",
        "google-generativeai>=0.3.0;python_version>='3.8'"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "flake8>=3.9.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "google-generativeai>=0.3.0"
        ]
    },
    author="AgentX Team",
    author_email="info@agentx.org",
    description="Multi-agent ethical decision making framework",
    keywords="ethics, agents, simulation, AI",
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "agentxthics-run=agentxthics.run_research:main",
            "agentxthics-simple=agentxthics.simple_run:main"
        ],
    },
)
