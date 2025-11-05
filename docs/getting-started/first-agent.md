# Your First Agent

Learn how to create your first SafeAgents agent in just a few minutes.

---

## What You'll Build

A simple weather agent that:
- Has a custom tool to get weather information
- Responds to natural language queries
- Can be run with any framework (Autogen, LangGraph, OpenAI Agents)

---

## Step 1: Create the Script

Create a new file `my_first_agent.py`:

```python
import asyncio
from safeagents import Agent, AgentConfig, Team, tool
from dotenv import load_dotenv

# Load API keys
load_dotenv()
```

---

## Step 2: Define a Tool

Tools are functions that agents can call. Let's create a weather tool:

```python
@tool()
def get_weather(city: str) -> str:
    """
    Get weather information for a city.

    Args:
        city: Name of the city

    Returns:
        Weather description
    """
    # In a real app, this would call a weather API
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 65°F",
        "London": "Rainy, 58°F",
        "Tokyo": "Clear, 68°F"
    }

    return weather_data.get(city, f"Weather data for {city} not available")
```

**Key Points:**
- `@tool()` decorator converts the function into a SafeAgents tool
- **Docstring is important!** The LLM sees it to understand what the tool does
- Type hints (`city: str -> str`) help the LLM use the tool correctly

---

## Step 3: Create an Agent

Now let's create an agent that can use this tool:

```python
# Create agent configuration
agent_config = AgentConfig(
    name="WeatherAgent",
    tools=[get_weather],
    system_message="You are a helpful weather assistant. Use the get_weather tool to answer questions about weather."
)

# Create the agent
agent = Agent(config=agent_config)
```

**What Each Parameter Does:**
- `name`: Unique identifier for the agent
- `tools`: List of tools the agent can use
- `system_message`: Instructions that guide the agent's behavior

---

## Step 4: Create a Team

Agents work within teams. Even a single agent needs a team:

```python
# Create a team with one agent
team = Team.create(
    agents=[agent],
    framework="openai-agents",  # Try: "autogen", "langgraph"
    architecture="centralized"
)
```

**Framework Options:**
- `"openai-agents"`: OpenAI's agent SDK
- `"autogen"`: Microsoft Autogen
- `"langgraph"`: LangChain's LangGraph

**Architecture Options:**
- `"centralized"`: One agent coordinates (best for single agents)
- `"decentralized"`: Agents coordinate among themselves (for multi-agent)

---

## Step 5: Run a Task

Now let's ask our agent a question:

```python
# Run a task
result = asyncio.run(team.run(
    task="What's the weather in San Francisco?",
    verbose=True  # Print logs during execution
))

# Print the result
print("\n" + "="*50)
print("RESULT:")
print("="*50)
print(result['logs'])
```

---

## Complete Code

Here's the full `my_first_agent.py`:

```python
import asyncio
from safeagents import Agent, AgentConfig, Team, tool
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Step 1: Define a tool
@tool()
def get_weather(city: str) -> str:
    """
    Get weather information for a city.

    Args:
        city: Name of the city

    Returns:
        Weather description
    """
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 65°F",
        "London": "Rainy, 58°F",
        "Tokyo": "Clear, 68°F"
    }

    return weather_data.get(city, f"Weather data for {city} not available")

# Step 2: Create an agent
agent = Agent(config=AgentConfig(
    name="WeatherAgent",
    tools=[get_weather],
    system_message="You are a helpful weather assistant. Use the get_weather tool to answer questions about weather."
))

# Step 3: Create a team
team = Team.create(
    agents=[agent],
    framework="openai-agents",
    architecture="centralized"
)

# Step 4: Run a task
result = asyncio.run(team.run(
    task="What's the weather in San Francisco?",
    verbose=True
))

# Print result
print("\n" + "="*50)
print("RESULT:")
print("="*50)
print(result['logs'])
```

---

## Run It!

```bash
python my_first_agent.py
```

**Expected Output:**

```
[SafeAgents] Creating openai-agents team with centralized architecture
[SafeAgents] Team created with 1 agent(s)
[SafeAgents] Running task...

[Agent: WeatherAgent] Analyzing task: "What's the weather in San Francisco?"
[Agent: WeatherAgent] Calling tool: get_weather(city="San Francisco")
[Tool: get_weather] Returned: "Sunny, 72°F"
[Agent: WeatherAgent] Formulating response...

==================================================
RESULT:
==================================================
The weather in San Francisco is Sunny, 72°F.
```

**🎉 Congratulations! You've created your first SafeAgents agent!**

---

## Understanding the Result

The `result` dictionary contains several useful fields:

```python
result = {
    'logs': str,                    # Full execution logs
    'messages': List[Dict],         # All messages exchanged
    'stop_reason': str,             # Why execution stopped
    'execution_trace': {            # Tool calls and commands
        'tool_calls': [...],
        'bash_commands': [...]
    },
    'task': str,                    # Original task
    'framework': str,               # Framework used
    'architecture': str             # Architecture used
}
```

**Access specific parts:**

```python
# Just the final answer
print(result['logs'])

# See what tools were called
for call in result['execution_trace']['tool_calls']:
    print(f"Called {call['name']} with args: {call['args']}")

# Check how it stopped
print(f"Stopped because: {result['stop_reason']}")
```

---

## Experiment: Try Different Queries

Modify the task to try different questions:

```python
# Try different cities
result = asyncio.run(team.run(
    task="What's the weather in London?",
    verbose=True
))

# Try multiple questions
result = asyncio.run(team.run(
    task="Compare the weather in New York and Tokyo",
    verbose=True
))

# Try an unknown city
result = asyncio.run(team.run(
    task="What's the weather in Paris?",
    verbose=True
))
```

---

## Experiment: Try Different Frameworks

Change just one line to use a different framework:

### OpenAI Agents (default)
```python
team = Team.create(
    agents=[agent],
    framework="openai-agents",
    architecture="centralized"
)
```

### Autogen
```python
team = Team.create(
    agents=[agent],
    framework="autogen",
    architecture="centralized"
)
```

### LangGraph
```python
team = Team.create(
    agents=[agent],
    framework="langgraph",
    architecture="centralized"
)
```

**The rest of the code stays exactly the same!** That's the power of SafeAgents.

---

## Adding More Tools

Let's make the agent more useful by adding a traffic tool:

```python
@tool()
def get_traffic(city: str) -> str:
    """
    Get traffic information for a city.

    Args:
        city: Name of the city

    Returns:
        Traffic description
    """
    traffic_data = {
        "San Francisco": "Moderate traffic, expect 20 min delays",
        "New York": "Heavy traffic, expect 45 min delays",
        "London": "Light traffic, clear roads",
        "Tokyo": "Heavy traffic, expect 30 min delays"
    }

    return traffic_data.get(city, f"Traffic data for {city} not available")

# Update agent with both tools
agent = Agent(config=AgentConfig(
    name="CityInfoAgent",  # Changed name
    tools=[get_weather, get_traffic],  # Added second tool
    system_message="You are a helpful assistant that provides weather and traffic information."
))

# Create team (same as before)
team = Team.create(
    agents=[agent],
    framework="openai-agents",
    architecture="centralized"
)

# Try a task that uses both tools
result = asyncio.run(team.run(
    task="What's the weather and traffic like in New York?",
    verbose=True
))
```

The agent will automatically figure out it needs to call both tools!

---

## Common Issues

### Issue: "Missing API key"

Make sure your `.env` file has:
```bash
OPENAI_API_KEY=sk-...your_key...
```

And you're loading it:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Issue: Agent doesn't use the tool

Check:
1. **Docstring**: Does the tool have a clear docstring?
2. **System message**: Does it mention the tool?
3. **Task**: Is the task clearly asking for something the tool provides?

---

## Next Steps

Now that you have a working agent, learn about:

1. **[Basic Concepts →](basic-concepts.md)** - Understand agents, tools, and teams deeply
2. **[Creating Agents →](../guides/creating-agents.md)** - Advanced agent configuration
3. **[Using Tools →](../guides/using-tools.md)** - Tool creation patterns
4. **[Multi-Agent Systems →](../examples/multi-agent-system.md)** - Multiple collaborating agents

---

## Challenge: Build Your Own

Try creating an agent for a different domain:

### Idea 1: Calculator Agent
```python
@tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Create agent with math tools...
```

### Idea 2: File Info Agent
```python
import os

@tool()
def list_files(directory: str) -> str:
    """List files in a directory."""
    return str(os.listdir(directory))

@tool()
def get_file_size(filepath: str) -> str:
    """Get size of a file in bytes."""
    return str(os.path.getsize(filepath))

# Create agent with file tools...
```

### Idea 3: Translation Agent
```python
@tool()
def translate_to_spanish(text: str) -> str:
    """Translate English text to Spanish."""
    translations = {
        "hello": "hola",
        "goodbye": "adiós",
        "thank you": "gracias"
    }
    return translations.get(text.lower(), f"Translation for '{text}' not found")

# Create agent with translation tools...
```

---

**🎉 You've mastered the basics! Keep exploring and building!**

---

[← Installation](installation.md) | [Basic Concepts →](basic-concepts.md)
