# Basic Concepts

Understanding the core concepts of SafeAgents.

---

## Overview

SafeAgents is built around four main concepts:

1. **Tools** - Functions that agents can call
2. **Agents** - Autonomous entities with tools and goals
3. **Teams** - Collections of agents working together
4. **Datasets** - Collections of tasks for evaluation

Let's explore each in detail.

---

## 1. Tools

### What is a Tool?

A **tool** is a function that an agent can call to perform an action or get information.

```python
from safeagents import tool

@tool()
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"Weather in {city}: Sunny, 72°F"
```

### Key Characteristics

**1. They're just Python functions**
```python
# Any function can become a tool
def my_function(x: int) -> int:
    return x * 2

# Use the @tool() decorator
@tool()
def my_tool(x: int) -> int:
    return x * 2
```

**2. The docstring matters**
```python
@tool()
def search(query: str) -> str:
    """
    Search the internet for information.

    Use this when you need up-to-date information
    that you don't already know.
    """
    return perform_search(query)
```

The LLM reads the docstring to understand when and how to use the tool!

**3. Type hints help**
```python
@tool()
def calculate(operation: str, a: float, b: float) -> float:
    """Perform arithmetic operations."""
    # Type hints tell the LLM what types to use
    ...
```

**4. Can be sync or async**
```python
# Synchronous
@tool()
def sync_tool(x: str) -> str:
    return process(x)

# Asynchronous
@tool()
async def async_tool(x: str) -> str:
    return await async_process(x)
```

SafeAgents handles both automatically!

### When Should You Create a Tool?

Create a tool when an agent needs to:
- **Access external systems** (APIs, databases, files)
- **Perform computations** (calculations, data processing)
- **Take actions** (send emails, create files, make requests)

**Examples:**
```python
@tool()
def read_file(path: str) -> str:
    """Read contents of a file."""
    with open(path) as f:
        return f.read()

@tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    # Email sending logic
    return f"Email sent to {to}"

@tool()
def query_database(sql: str) -> str:
    """Execute SQL query on database."""
    # Database query logic
    return str(results)
```

---

## 2. Agents

### What is an Agent?

An **agent** is an autonomous entity that:
- Has access to **tools**
- Follows a **system message** (instructions)
- Can **hand off** tasks to other agents (in case of decentralized architecture)
- Makes **decisions** about which tools to use

```python
from safeagents import Agent, AgentConfig

agent = Agent(config=AgentConfig(
    name="WeatherAgent",
    tools=[get_weather],
    system_message="You provide weather information.",
    description="An agent that knows about weather.",
    handoffs=["TrafficAgent"]
))
```

### Agent Configuration

#### Required Parameters

**`name`** - Unique identifier
```python
name="WeatherAgent"  # Must be unique within a team
```

#### Optional Parameters

**`tools`** - List of tools the agent can use
```python
tools=[get_weather, get_forecast]
```

**`system_message`** - Instructions for the agent
```python
system_message="You are a helpful weather assistant. Always use the get_weather tool to provide accurate information."
```

**`description`** - What the agent does
```python
description="An agent that provides weather information for cities worldwide."
```

**`handoffs`** - Names of agents this agent can delegate to (applicable in decentralized teams)
```python
handoffs=["TrafficAgent", "RestaurantAgent"]
```

**`handoff_description`** - When other agents should delegate to this agent (applicable in decentralized teams)
```python
handoff_description="Transfer to WeatherAgent for questions about weather, temperature, or climate."
```

### Special Agents

SafeAgents provides pre-built agents for common tasks:

```python
# File operations agent
file_agent = Agent(config=AgentConfig(
    name="FileSurfer",
    special_agent="file_surfer"
))

# Web browsing agent
web_agent = Agent(config=AgentConfig(
    name="WebSurfer",
    special_agent="web_surfer"
))

# Code execution agent
coder_agent = Agent(config=AgentConfig(
    name="Coder",
    special_agent="coder"
))

# Terminal/bash agent
terminal_agent = Agent(config=AgentConfig(
    name="ComputerTerminal",
    special_agent="computer_terminal"
))
```

**Note:** Special agents cannot have custom tools.

### Agent Behavior

Agents make decisions based on:
1. **System message** - Their instructions
2. **Available tools** - What they can do
3. **Conversation history** - Previous messages
4. **Task** - The current goal

Example:
```python
agent = Agent(config=AgentConfig(
    name="ResearchAgent",
    tools=[web_search, read_article, summarize],
    system_message="""
    You are a research assistant. When given a topic:
    1. Search for relevant articles using web_search
    2. Read the best articles using read_article
    3. Summarize findings using summarize
    """
))
```

The agent will follow these instructions when executing tasks.

---

## 3. Teams

### What is a Team?

A **team** is a collection of agents that work together to accomplish tasks.

```python
from safeagents import Team

team = Team.create(
    agents=[weather_agent, traffic_agent],
    framework="autogen",
    architecture="centralized"
)
```

### Team Configuration

#### Required Parameters

**`agents`** - List of agents in the team
```python
agents=[agent1, agent2, agent3]
```

**`framework`** - Which execution framework to use
```python
framework="autogen"  # or "langgraph", "openai-agents"
```

**`architecture`** - How agents coordinate
```python
architecture="centralized"  # or "decentralized"
```

#### Optional Parameters

**`max_turns`** - Maximum conversation turns
```python
max_turns=15  # Default: 10
```

Prevents infinite loops.

### Frameworks

SafeAgents supports three frameworks:

| Framework | Description | Best For |
|-----------|-------------|----------|
| **autogen** | Microsoft Autogen | Complex multi-agent workflows |
| **langgraph** | LangChain LangGraph | Graph-based agent coordination |
| **openai-agents** | OpenAI Agents SDK | Simple, single-agent tasks |

**Switch frameworks without changing code:**
```python
# Same agents, different framework
team1 = Team.create(agents=[agent], framework="autogen", ...)
team2 = Team.create(agents=[agent], framework="langgraph", ...)
team3 = Team.create(agents=[agent], framework="openai-agents", ...)
```

### Architectures

#### Centralized Architecture

One agent (or coordinator) manages the workflow:

```
        [Coordinator]
          /    |    \
         /     |     \
    [Agent1] [Agent2] [Agent3]
```

**Best for:**
- Single-agent tasks
- Clear task delegation
- Simpler workflows

```python
team = Team.create(
    agents=[agent],
    framework="openai-agents",
    architecture="centralized"
)
```

#### Decentralized Architecture

Agents communicate directly with each other:

```
    [Agent1] <---> [Agent2]
       ^              ^
       |              |
       v              v
    [Agent3] <---> [Agent4]
```

**Best for:**
- Multi-agent collaboration
- Complex negotiations
- Dynamic workflows

```python
team = Team.create(
    agents=[agent1, agent2, agent3],
    framework="autogen",
    architecture="decentralized"
)
```

### Running Tasks

Execute tasks with `team.run()`:

```python
import asyncio

result = asyncio.run(team.run(
    task="What's the weather in NYC?",
    verbose=True,  # Print logs
    assessment=["aria"],  # Run safety assessment
    attack_detector=my_detector  # Detect attacks
))
```

**Parameters:**
- `task` (str): The task to accomplish
- `verbose` (bool): Print execution logs
- `assessment` (List[str]): Safety metrics to run (["aria", "dharma"])
- `attack_detector` (Callable): Function to detect attacks

**Returns:** Dictionary with:
```python
{
    'logs': str,  # Execution logs
    'messages': List[Dict],  # All messages
    'stop_reason': str,  # Why it stopped
    'execution_trace': {...},  # Tool calls
    'assessment': {...},  # ARIA/DHARMA scores (if requested)
    'attack_detected': bool  # True if attack found
}
```

---

## 4. Datasets

### What is a Dataset?

A **dataset** is a collection of tasks for evaluation and benchmarking.

```python
from safeagents import Dataset

dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",
    framework="openai-agents",
    architecture="centralized",
    indices=[0, 1, 2]
).load()

results = dataset.run(assessment=["aria", "dharma"])
```

### Supported Datasets

#### AgentHarm
AI safety benchmark with harmful and benign tasks.

```python
dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",  # or "harmless_benign", "chat"
    split="test_public",  # or "validation"
    framework="autogen",
    architecture="centralized",
    indices=[0, 1, 2]  # Run specific tasks
).load()
```

#### ASB (Agent Security Benchmark)
Comprehensive agent safety evaluation.

```python
dataset = Dataset(
    name="asb",
    config="financial_analyst_agent",  # Specific agent name
    framework="langgraph",
    architecture="decentralized",
    indices=[0, 1, 2]
).load()
```

### Dataset Features

#### Checkpointing
Resume interrupted runs:

```python
# First run
dataset.run(resume=True, indices=range(100))
# ... interrupted ...

# Resume later
dataset.run(resume=True, indices=range(100))
# Skips completed tasks!
```

#### Progress Tracking
```python
results = dataset.run(
    progress_bar=True,  # Show progress bar
    save_checkpoints=True  # Save after each task
)
```

#### Summary Statistics
```python
# After running
dataset.print_summary()
```

**Output:**
```
================================================================================
DATASET RUN SUMMARY
================================================================================
Total tasks: 10
Successful: 9
Errors: 1

   ARIA Score Distribution
┏━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Score ┃ Count ┃ Percentage ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│   1   │     5 │      55.6% │
│   2   │     2 │      22.2% │
│   4   │     2 │      22.2% │
└───────┴───────┴────────────┘
```

---

## Putting It All Together

Here's how all concepts work together:

```python
import asyncio
from safeagents import Agent, AgentConfig, Team, tool

# 1. Create tools
@tool()
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny"

@tool()
def get_traffic(city: str) -> str:
    """Get traffic for a city."""
    return f"Traffic in {city}: Light"

# 2. Create agents with tools
weather_agent = Agent(config=AgentConfig(
    name="WeatherAgent",
    tools=[get_weather],
    system_message="Provide weather information.",
    handoffs=["TrafficAgent"]
))

traffic_agent = Agent(config=AgentConfig(
    name="TrafficAgent",
    tools=[get_traffic],
    system_message="Provide traffic information.",
    handoffs=["WeatherAgent"]
))

# 3. Create a team
team = Team.create(
    agents=[weather_agent, traffic_agent],
    framework="autogen",
    architecture="decentralized",
    max_turns=10
)

# 4. Run a task
result = asyncio.run(team.run(
    task="What's the weather and traffic in San Francisco?",
    verbose=True
))

print(result['logs'])
```

**What happens:**
1. Task is given to the team
2. Agents decide who should handle it
3. WeatherAgent calls `get_weather("San Francisco")`
4. TrafficAgent calls `get_traffic("San Francisco")`
5. Agents collaborate to form final response
6. Result is returned

---

## Key Takeaways

### Tools
✅ Functions that agents call to perform actions
✅ Decorated with `@tool()`
✅ Docstrings teach the LLM when to use them

### Agents
✅ Autonomous entities with tools and instructions
✅ Can hand off tasks to other agents
✅ Configured with `AgentConfig`

### Teams
✅ Collections of agents working together
✅ Support multiple frameworks (autogen, langgraph, openai-agents)
✅ Use centralized or decentralized architectures

### Datasets
✅ Collections of tasks for benchmarking
✅ Support checkpointing and progress tracking
✅ Provide summary statistics

---

## Next Steps

Now that you understand the basics:

- **[Creating Agents →](../guides/creating-agents.md)** - Advanced agent patterns
- **[Using Tools →](../guides/using-tools.md)** - Tool creation best practices
- **[Running Datasets →](../guides/running-datasets.md)** - Benchmark execution
- **[Examples →](../examples/)** - See real-world applications

---

[← First Agent](first-agent.md) | [Creating Agents →](../guides/creating-agents.md)
