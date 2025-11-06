# SafeAgents ( also called `SafeAgentEval` )

> **A unified framework for building and evaluating safe multi-agent systems**

SafeAgents provides a simple, framework-agnostic API for creating multi-agent systems with built-in safety evaluation, attack detection, and support for multiple agentic frameworks (Autogen, LangGraph, OpenAI Agents).

---

## ✨ Key Features

- 🤖 **Multi-Framework Support**: Write once, run on Autogen, LangGraph, or OpenAI Agents
- 🏗️ **Multiple Architectures**: Centralized or decentralized agent coordination
- 🛡️ **Built-in Safety**: Attack detection and safety evaluation (ARIA, DHARMA)
- 🔧 **Special Agents**: Pre-built agents for web browsing, file operations, and code execution
- 📊 **Dataset Support**: Run benchmarks like AgentHarm and ASB with checkpointing
- 🔄 **Agent Handoffs**: Seamless task delegation between agents
- 📈 **Progress Tracking**: Checkpoint/resume for long-running experiments

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SafeAgentEval.git
cd SafeAgents

# Create environment (choose one)
# Option 1: Using conda
conda create -n safeagents python=3.12
conda activate safeagents

# Option 2: Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install Playwright for web_surfer
playwright install --with-deps chromium
```

### Your First Agent (30 seconds)

```python
import asyncio
from safeagents import Agent, AgentConfig, Team, tool

# Define a tool
@tool()
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"Weather in {city}: Sunny and 72°F"

# Create an agent
agent = Agent(config=AgentConfig(
    name="WeatherAgent",
    tools=[get_weather],
    system_message="You are a helpful weather assistant."
))

# Create a team
team = Team.create(
    agents=[agent],
    framework="openai-agents",  # or "autogen", "langgraph"
    architecture="centralized"
)

# Run a task
result = asyncio.run(team.run(
    task="What's the weather in San Francisco?",
    verbose=True
))

print(result['logs'])
```

**Output:**
```
Weather in San Francisco: Sunny and 72°F
```

---

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get up and running in 5 minutes
- **[Getting Started](docs/getting-started/)** - Tutorials and core concepts
- **[Feature Guides](docs/guides/)** - In-depth guides for each feature
- **[Examples](docs/examples/)** - Real-world usage examples

---

## 🎯 Use Cases

### 1. Multi-Agent Collaboration

```python
# Create specialized agents that can hand off tasks
weather_agent = Agent(config=AgentConfig(
    name="WeatherAgent",
    tools=[get_weather],
    handoffs=["TrafficAgent"]  # Can delegate to TrafficAgent
))

traffic_agent = Agent(config=AgentConfig(
    name="TrafficAgent",
    tools=[get_traffic],
    handoffs=["WeatherAgent"]
))

team = Team.create(
    agents=[weather_agent, traffic_agent],
    framework="autogen",
    architecture="decentralized"
)

result = asyncio.run(team.run(
    "What's the weather and traffic in NYC?"
))
```

### 2. Safety Evaluation on Benchmarks

```python
from safeagents import Dataset

# Load AgentHarm benchmark
dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",
    framework="openai-agents",
    architecture="centralized",
    indices=[0, 1, 2]  # Run first 3 tasks
).load()

# Run with automatic safety assessment
results = dataset.run(
    assessment=["aria", "dharma"],
    progress_bar=True
)

# View summary with score distributions
dataset.print_summary()
```

**Output:**
```
================================================================================
DATASET RUN SUMMARY
================================================================================
Total tasks: 3
Successful: 3
Errors: 0

   ARIA Score Distribution
┏━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Score ┃ Count ┃ Percentage ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│   1   │     2 │      66.7% │
│   4   │     1 │      33.3% │
└───────┴───────┴────────────┘
```

### 3. Attack Detection

```python
from safeagents.core.src.evaluation.attack_detection import tools_called, any_of

# Detect if dangerous tools are called
detector = any_of(
    tools_called(['delete_file']),
    tools_called(['send_email'])
)

result = asyncio.run(team.run(
    task="Delete sensitive files",
    attack_detector=detector,
    assessment=["aria"]
))

if result['attack_detected']:
    print(f"🚨 Attack detected! ARIA: {result['assessment']['aria']}")
```

### 4. Special Agents

```python
# Use pre-built agents for common tasks
file_agent = Agent(config=AgentConfig(
    name="FileSurfer",
    special_agent="file_surfer"  # Built-in file operations
))

web_agent = Agent(config=AgentConfig(
    name="WebSurfer",
    special_agent="web_surfer"  # Built-in web browsing
))

team = Team.create(
    agents=[file_agent, web_agent],
    framework="langgraph",
    architecture="centralized"
)
```

---

## 🔧 Supported Frameworks

| Framework | Status | Architecture Support |
|-----------|--------|---------------------|
| **Autogen** | ✅ Fully Supported | Centralized, Decentralized |
| **LangGraph** | ✅ Fully Supported | Centralized, Decentralized |
| **OpenAI Agents** | ✅ Fully Supported | Centralized only |

---

## 📊 Supported Datasets

| Dataset | Description | Config Options |
|---------|-------------|----------------|
| **AgentHarm** | AI safety benchmark with harmful tasks | `harmful`, `harmless_benign`, `chat` |
| **ASB** | Agent Safety Benchmark | Agent-specific configs |
| **Custom** | Bring your own dataset | Create a dataset handler |

**See [Dataset Guide](docs/guides/running-datasets.md) for more details.**

---

## 🛡️ Safety Features

### Attack Detection
Detect malicious behavior during execution:
- Tool call monitoring
- Bash command tracking
- Log pattern matching
- Custom detection logic

### Assessment Metrics
- **[ARIA](https://arxiv.org/abs/2503.04957)**: Agent Risk Assessment for AI systems
- **DHARMA**: Domain-specific Harm Assessment (Design aware Harm Assessment Metric for Agents)
- Automatic ARIA=4 assignment when attacks are detected

**See [Attack Detection Guide](docs/guides/attack-detection.md) for details.**

---

## 📖 Core Concepts

### Agent
An autonomous entity with tools and capabilities.

```python
agent = Agent(config=AgentConfig(
    name="MyAgent",
    tools=[my_tool],
    system_message="You are a helpful assistant.",
    handoffs=["OtherAgent"]  # Can delegate to other agents
))
```

### Tool
A function that agents can call to perform actions.

```python
@tool()
def my_tool(input: str) -> str:
    """Tool description for the LLM."""
    return f"Processed: {input}"
```

### Team
A collection of agents working together.

```python
team = Team.create(
    agents=[agent1, agent2],
    framework="autogen",
    architecture="centralized",
    max_turns=10
)
```

### Dataset
Run benchmarks or experiments across multiple tasks.

```python
dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    framework="openai-agents",
    architecture="centralized"
).load()

results = dataset.run(assessment=["aria", "dharma"])
```

---

## 🗂️ Project Structure

```
SafeAgents/
├── safeagents/
│   ├── core/                  # Core framework code
│   │   └── src/
│   │       ├── models/        # Agent, Tool, Task models
│   │       ├── frameworks/    # Framework implementations
│   │       ├── evaluation/    # ARIA, DHARMA, attack detection
│   │       └── datasets/      # Dataset management
│   └── datasets/              # Dataset handlers
│       ├── agentharm/         # AgentHarm handler
│       └── asb/               # ASB handler
├── docs/                      # Documentation
├── example_scripts/           # Working examples
└── README.md                  # This file
```

---

## 🌟 Why SafeAgents?

### Before SafeAgents
```python
# Different code for each framework
if framework == "autogen":
    # Autogen-specific code
    from autogen import AssistantAgent
    agent = AssistantAgent(...)
elif framework == "langgraph":
    # LangGraph-specific code
    from langgraph import Agent
    agent = Agent(...)
# ... more framework-specific code
```

### With SafeAgents
```python
# One API, multiple frameworks
from safeagents import Agent, Team

agent = Agent(config=AgentConfig(...))
team = Team.create(
    agents=[agent],
    framework="autogen"  # Just change this!
)
```

**Switch frameworks without rewriting code!**


---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Autogen](https://github.com/microsoft/autogen) - Multi-agent framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Graph-based agent orchestration
- [OpenAI Agents](https://platform.openai.com/docs/) - OpenAI's agent SDK
- [AgentHarm](https://arxiv.org/abs/2410.09024) - Safety benchmark
- [ASB](https://arxiv.org/abs/2410.02644) - Agent Security Benchmark

---

## 📬 Contact

For questions, issues, or feedback:
- **Issues**: [GitHub Issues](https://github.com/yourusername/SafeAgentEval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/SafeAgentEval/discussions)

---

## 🚦 Quick Links

- [Quick Start Guide →](QUICKSTART.md)
- [Installation Guide →](docs/getting-started/installation.md)
- [Your First Agent →](docs/getting-started/first-agent.md)
- [API Reference →](docs/api-reference/)
- [Examples →](docs/examples/)

---

<p align="center">
  <strong>Built with ❤️ for safe AI systems</strong>
</p>

## Trademark Notice

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.