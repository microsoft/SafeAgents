## SafeAgents Examples Index

Quick reference guide to all available examples.

---

### 🚀 Quick Start

**Start here if you're new to SafeAgents!**

| Example | Description | Key Features | Run Time |
|---------|-------------|--------------|----------|
| [00_hello_world.py](00_hello_world.py) | Simplest possible example | Basic agent, tool, and task | < 30 sec |

```bash
python 00_hello_world.py
```

---

### 📚 Core Examples

| # | Example | Description | Difficulty |
|---|---------|-------------|------------|
| 01 | [basic_multi_agent.py](01_basic_multi_agent.py) | Multi-agent system fundamentals | ⭐ Beginner |
| 02 | [attack_detection.py](02_attack_detection.py) | Detecting harmful behavior | ⭐⭐ Intermediate |
| 03 | [dataset_evaluation.py](03_dataset_evaluation.py) | Running safety benchmarks | ⭐⭐ Intermediate |
| 04 | [special_agents.py](04_special_agents.py) | Using built-in special agents | ⭐⭐⭐ Advanced |
| 05 | [docker_isolation.py](05_docker_isolation.py) | Running tasks in Docker containers | ⭐⭐ Intermediate |

---

### 📖 By Use Case

#### Building Multi-Agent Systems
- **[01_basic_multi_agent.py](01_basic_multi_agent.py)** - Start here
  - Creating agents with custom tools
  - Agent handoffs and collaboration
  - Framework selection (openai-agents, autogen, langgraph)

#### Safety & Security
- **[02_attack_detection.py](02_attack_detection.py)** - Comprehensive safety
  - Tool-based detection
  - Custom detection functions
  - Combining multiple detectors

#### Research & Evaluation
- **[03_dataset_evaluation.py](03_dataset_evaluation.py)** - Benchmark evaluation
  - AgentHarm dataset
  - ASB (Agent Safety Benchmark)
  - Framework comparisons

#### Advanced Capabilities
- **[04_special_agents.py](04_special_agents.py)** - Powerful agents
  - File system operations
  - Terminal/bash execution
  - Safety best practices

#### Isolation & Safety
- **[05_docker_isolation.py](05_docker_isolation.py)** - Docker containers
  - Running tasks in isolation
  - Saving results from containers
  - Multiple task execution
  - Custom package installation

---

### 🎯 By Framework

#### OpenAI Agents
```bash
python 01_basic_multi_agent.py --framework openai-agents
```
- Requires: OpenAI API key
- Best for: Production use
- Features: Native assistant API, streaming

#### Autogen
```bash
python 01_basic_multi_agent.py --framework autogen
```
- Requires: Azure OpenAI
- Best for: Research, special agents
- Features: Conversational, flexible

#### LangGraph
```bash
python 01_basic_multi_agent.py --framework langgraph
```
- Requires: Azure OpenAI
- Best for: Complex workflows
- Features: Graph-based, state management

---

### 🔍 By Feature

#### Assessment (ARIA & DHARMA)
All examples support assessment:
```python
result = await team.run(
    task="...",
    assessment=["aria", "dharma"]
)
```

**Examples using assessment:**
- ✅ [01_basic_multi_agent.py](01_basic_multi_agent.py) - Line 123
- ✅ [02_attack_detection.py](02_attack_detection.py) - Multiple examples
- ✅ [03_dataset_evaluation.py](03_dataset_evaluation.py) - All examples
- ✅ [04_special_agents.py](04_special_agents.py) - Line 176

#### Attack Detection
**Examples:**
- 🎯 [02_attack_detection.py](02_attack_detection.py) - **Main focus**
  - Example 1: Simple tool detection
  - Example 2: Combining detectors
  - Example 3: Custom detector
  - Example 4: Bash detection
  - Example 5: AND logic (all_of)

- 🎯 [04_special_agents.py](04_special_agents.py) - Example 4
  - Multi-layered detection for special agents

#### Dataset Evaluation
**Examples:**
- 📊 [03_dataset_evaluation.py](03_dataset_evaluation.py) - **Main focus**
  - Example 1: Basic run
  - Example 2: Checkpoint/resume
  - Example 3: Framework comparison
  - Example 4: Custom subset
  - Example 5: Result analysis
  - Example 6: ASB dataset

#### Special Agents
**Examples:**
- 🔧 [04_special_agents.py](04_special_agents.py) - **Main focus**
  - Example 1: file_surfer
  - Example 2: computer_terminal
  - Example 3: Combined agents
  - Example 4: Safety evaluation

---

### ⚡ Quick Commands

```bash
# Hello World - Verify installation
python 00_hello_world.py

# Basic multi-agent system
python 01_basic_multi_agent.py

# Run with different framework
python 01_basic_multi_agent.py --framework autogen

# Attack detection examples
python 02_attack_detection.py              # All examples
python 02_attack_detection.py --example 3  # Custom detector

# Dataset evaluation
python 03_dataset_evaluation.py 1    # Basic run
python 03_dataset_evaluation.py 3    # Framework comparison

# Special agents
python 04_special_agents.py              # All examples
python 04_special_agents.py --example 1  # File surfer
```

---

### 📊 Example Comparison

| Feature | 00_hello | 01_basic | 02_attack | 03_dataset | 04_special | 05_docker |
|---------|----------|----------|-----------|------------|------------|-----------|
| **Difficulty** | ⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Run Time** | < 30s | 1-2 min | 2-3 min | 5-10 min | 2-3 min | 3-5 min |
| **Custom Tools** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Multiple Agents** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Assessment** | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Attack Detection** | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| **Dataset Support** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Special Agents** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Docker Isolation** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

### 💡 Learning Path

**Beginner → Advanced**

1. **Start**: [00_hello_world.py](00_hello_world.py)
   - Verify installation
   - Understand basic structure

2. **Learn**: [01_basic_multi_agent.py](01_basic_multi_agent.py)
   - Create custom tools
   - Build multi-agent systems
   - Try different frameworks

3. **Safety**: [02_attack_detection.py](02_attack_detection.py)
   - Understand attack detection
   - Implement safety checks
   - Custom detection logic

4. **Evaluate**: [03_dataset_evaluation.py](03_dataset_evaluation.py)
   - Run benchmarks
   - Compare frameworks
   - Analyze results

5. **Advanced**: [04_special_agents.py](04_special_agents.py)
   - Use powerful capabilities
   - Combine agent types
   - Production safety practices

---

### 🔗 Related Documentation

- **Guides**: [docs/guides/](../guides/)
  - [Getting Started](../guides/getting-started.md)
  - [Creating Agents](../guides/creating-agents.md)
  - [Assessment (ARIA & DHARMA)](../guides/assessment.md)
  - [Attack Detection](../guides/attack-detection.md)


---

### ❓ Need Help?

- **Issue**: Example not working? [Report it](https://github.com/microsoft/SafeAgentEval/issues)
- **Question**: Need clarification? [Discussions](https://github.com/your-org/SafeAgentEval/discussions)
- **Docs**: Check [README](README.md) for troubleshooting

---

**Last Updated**: 2025-11-03
