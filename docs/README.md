# SafeAgents Documentation

Welcome to the SafeAgents documentation!

---

## 🚀 Getting Started

New to SafeAgents? Start here:

1. **[Installation](getting-started/installation.md)** - Set up SafeAgents
2. **[Quick Start](../QUICKSTART.md)** - Get running in 5 minutes
3. **[Your First Agent](getting-started/first-agent.md)** - Create your first agent
4. **[Basic Concepts](getting-started/basic-concepts.md)** - Understand the framework

---

## 📚 Feature Guides

In-depth guides for each feature:

### Core Features
- **[Creating Agents](guides/creating-agents.md)** - Agent configuration and patterns
- **[Using Tools](guides/using-tools.md)** - Tool creation and best practices
- **[Teams & Frameworks](guides/teams-and-frameworks.md)** - Multi-agent coordination
- **[Special Agents](guides/special-agents.md)** - Pre-built agents (web_surfer, file_surfer, etc.)

### Evaluation & Safety
- **[Running Datasets](guides/running-datasets.md)** - Benchmarks and checkpointing
- **[Assessment (ARIA/DHARMA)](guides/assessment.md)** - Safety evaluation metrics
- **[Attack Detection](guides/attack-detection.md)** - Detecting harmful behavior

### Advanced
- **[Architectures](guides/architectures.md)** - Centralized vs Decentralized
- **[Custom Datasets](guides/custom-datasets.md)** - Creating dataset handlers
- **[Docker Isolation](guides/docker-isolation.md)** - Running in containers (coming soon)

---

## 📖 API Reference

Complete API documentation:

- **[Agent API](api-reference/agent.md)** - Agent, AgentConfig
- **[Tool API](api-reference/tool.md)** - Tool class, @tool decorator
- **[Team API](api-reference/team.md)** - Team, Framework, Architecture
- **[Dataset API](api-reference/dataset.md)** - Dataset, DatasetRegistry
- **[Evaluation API](api-reference/evaluation.md)** - Assessment functions

---

## 💡 Examples

Real-world usage examples:

- **[Simple Weather Agent](examples/simple-weather-agent.md)** - Basic agent example
- **[Multi-Agent System](examples/multi-agent-system.md)** - Agent collaboration
- **[Benchmark Evaluation](examples/benchmark-evaluation.md)** - Running experiments
- **[Attack Detection Example](examples/attack-detection-example.md)** - Security features

---

## 🎯 Quick Reference

### Creating an Agent

```python
from safeagents import Agent, AgentConfig, tool

@tool()
def my_tool(input: str) -> str:
    """Tool description."""
    return f"Processed: {input}"

agent = Agent(config=AgentConfig(
    name="MyAgent",
    tools=[my_tool],
    system_message="You are a helpful assistant."
))
```

### Running a Task

```python
import asyncio
from safeagents import Team

team = Team.create(
    agents=[agent],
    framework="openai-agents",
    architecture="centralized"
)

result = asyncio.run(team.run(
    task="Your task here",
    verbose=True
))
```

### Running a Dataset

```python
from safeagents import Dataset

dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",
    framework="openai-agents",
    architecture="centralized",
    indices=[0, 1, 2]
).load()

results = dataset.run(
    assessment=["aria", "dharma"],
    progress_bar=True
)

dataset.print_summary()
```

---

## 🔍 Search by Topic

### By Task

- **Creating agents** → [Creating Agents](guides/creating-agents.md), [First Agent](getting-started/first-agent.md)
- **Adding tools** → [Using Tools](guides/using-tools.md)
- **Multi-agent systems** → [Teams & Frameworks](guides/teams-and-frameworks.md)
- **Running benchmarks** → [Running Datasets](guides/running-datasets.md)
- **Safety evaluation** → [Assessment](guides/assessment.md), [Attack Detection](guides/attack-detection.md)
- **Web browsing** → [Special Agents](guides/special-agents.md)
- **File operations** → [Special Agents](guides/special-agents.md)

### By Framework

- **Autogen** → [Teams & Frameworks](guides/teams-and-frameworks.md)
- **LangGraph** → [Teams & Frameworks](guides/teams-and-frameworks.md)
- **OpenAI Agents** → [Teams & Frameworks](guides/teams-and-frameworks.md)

### By Feature

- **Checkpointing** → [Running Datasets](guides/running-datasets.md)
- **Progress tracking** → [Running Datasets](guides/running-datasets.md)
- **Agent handoffs** → [Creating Agents](guides/creating-agents.md)
- **Attack detection** → [Attack Detection](guides/attack-detection.md)
- **ARIA scores** → [Assessment](guides/assessment.md)
- **DHARMA scores** → [Assessment](guides/assessment.md)

---

## 🐛 Troubleshooting

### Common Issues

**Import errors:**
```python
# ✅ Correct (new style)
from safeagents import Agent, Team, Dataset

# ❌ Deprecated (old style)
from safeagents.core.src import Agent, Team, Dataset
```

**Missing API keys:**
- Create `.env` file with `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY`
- Use `load_dotenv()` in your script

**Playwright not installed:**
```bash
playwright install chromium
```

**See full troubleshooting:** [Installation Guide](getting-started/installation.md#troubleshooting)

---

## 🤝 Contributing

Want to contribute? Check out:
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [GitHub Issues](https://github.com/yourusername/SafeAgents/issues) - Report bugs or request features

---

## 📬 Getting Help

- **Documentation**: You're here! Use the search or browse by topic
- **Examples**: Check [examples/](examples/) for working code
- **Issues**: [GitHub Issues](https://github.com/yourusername/SafeAgents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/SafeAgents/discussions)

---

## 📝 Documentation Structure

```
docs/
├── getting-started/           # Tutorials for beginners
│   ├── installation.md
│   ├── first-agent.md
│   └── basic-concepts.md
├── guides/                    # Feature-specific guides
│   ├── creating-agents.md
│   ├── using-tools.md
│   ├── running-datasets.md
│   ├── attack-detection.md
│   └── assessment.md
├── api-reference/             # Complete API docs
│   ├── agent.md
│   ├── tool.md
│   ├── team.md
│   └── dataset.md
└── examples/                  # Real-world examples
    ├── simple-weather-agent.md
    ├── multi-agent-system.md
    └── benchmark-evaluation.md
```

---

[← Back to Main README](../README.md) | [Installation →](getting-started/installation.md)
