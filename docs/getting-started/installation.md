# Installation Guide

Complete instructions for installing and configuring SafeAgents.

---

## Prerequisites

### Required
- **Python 3.8 or higher**
- **Conda** (recommended) or **pip**
- **Git**

---

## Installation Methods

### Method 1: Conda (Recommended)

Conda provides better isolation and environment management.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/SafeAgents.git
cd SafeAgents

# 2. Create conda environment
conda create -n safeagents python=3.10

# 3. Activate environment
conda activate safeagents

# 4. Install requirements
pip install -r requirements.txt

# 5. Install Playwright for web_surfer (optional)
playwright install chromium

# 6. Verify installation
python -c "from safeagents import Agent, Team; print('✅ SafeAgents installed successfully!')"
```

### Method 2: pip + virtualenv

If you prefer pip:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/SafeAgents.git
cd SafeAgents

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Install Playwright (optional)
playwright install chromium

# 5. Verify installation
python -c "from safeagents import Agent, Team; print('✅ SafeAgents installed successfully!')"
```

---

## API Configuration

SafeAgents requires API keys for the underlying LLM providers.

### Step 1: Create `.env` File

Create a file named `.env` in the project root:

```bash
touch .env
```

### Step 2: Add API Keys

#### Option A: OpenAI

```bash
# .env
OPENAI_API_KEY=sk-...your_key_here...
```

#### Option B: Azure OpenAI

```bash
# .env
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4

# Optional: Specific deployments for different models
AZURE_GPT4_DEPLOYMENT=gpt-4
AZURE_GPT35_DEPLOYMENT=gpt-35-turbo
```

#### Option C: Both

```bash
# .env
# OpenAI
OPENAI_API_KEY=sk-...your_key_here...

# Azure OpenAI (fallback)
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4
```

### Step 3: Load Environment Variables

Your scripts should load the `.env` file:

```python
from dotenv import load_dotenv

load_dotenv()  # Loads .env file from current directory
```

**Security Note:** Never commit `.env` to version control! It's already in `.gitignore`.

---

## Optional Components

### Playwright (for web_surfer)

If you want to use the `web_surfer` special agent:

```bash
# Install Playwright
conda activate safeagents
playwright install chromium

# Verify
playwright --version
```

### Rich (for pretty tables)

Enhanced summary output with colored tables:

```bash
pip install rich
```

Without `rich`, summaries will use plain ASCII tables (still functional).

---

## Framework-Specific Setup

SafeAgents supports multiple frameworks. Each has specific dependencies:

### Autogen

Already included in requirements. No additional setup needed.

```python
from safeagents import Team

team = Team.create(
    agents=[...],
    framework="autogen"
)
```

### LangGraph

Already included in requirements. No additional setup needed.

```python
from safeagents import Team

team = Team.create(
    agents=[...],
    framework="langgraph"
)
```

### OpenAI Agents

Already included in requirements. Requires OpenAI API key.

```python
from safeagents import Team

team = Team.create(
    agents=[...],
    framework="openai-agents"
)
```

---

## Verifying Installation

### Test 1: Import SafeAgents

```bash
python -c "from safeagents import Agent, Team, Dataset, tool; print('✅ All imports successful')"
```

**Expected Output:**
```
✅ All imports successful
```

### Test 2: Run Simple Agent

Create `test_install.py`:

```python
import asyncio
from safeagents import Agent, AgentConfig, Team, tool

@tool()
def test_tool(input: str) -> str:
    """A test tool."""
    return f"Processed: {input}"

agent = Agent(config=AgentConfig(
    name="TestAgent",
    tools=[test_tool],
    system_message="You are a test assistant."
))

team = Team.create(
    agents=[agent],
    framework="openai-agents",
    architecture="centralized"
)

result = asyncio.run(team.run(
    task="Use the test tool with input 'hello'",
    verbose=True
))

print("\n✅ Installation verified! SafeAgents is working correctly.")
```

Run it:

```bash
python test_install.py
```

If this works, you're all set!

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'safeagents'`

**Solution:**

```bash
# Make sure you're in the SafeAgents directory
cd /path/to/SafeAgents

# Make sure conda environment is activated
conda activate safeagents

# Verify Python is from conda environment
which python  # Should show conda path

# Try import again
python -c "from safeagents import Agent"
```


### Issue: `Missing API key`

**Solution:**

```bash
# Check if .env file exists
ls -la .env

# Check if API key is loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

If it prints `None`, your `.env` file is not being loaded or doesn't have the key.

### Issue: `Playwright browser not found`

**Solution:**

```bash
# Install Playwright browsers
playwright install chromium

# Or install all browsers
playwright install
```

### Issue: `Conda environment creation fails`

**Solution:**

```bash
# Try updating conda first
conda update conda

# Try creating environment with specific Python version
conda create -n safeagents python=3.10
conda activate safeagents
pip install -r requirements.txt
```


---

## Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes* | OpenAI API key | `sk-...` |
| `AZURE_OPENAI_API_KEY` | Yes* | Azure OpenAI API key | `abc123...` |
| `AZURE_OPENAI_ENDPOINT` | No | Azure endpoint URL | `https://resource.openai.azure.com` |
| `AZURE_OPENAI_API_VERSION` | No | Azure API version | `2024-02-15-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | No | Azure deployment name | `gpt-4` |

*At least one of `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` is required.

---

## Development Installation

If you want to contribute or modify SafeAgents:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/SafeAgents.git
cd SafeAgents

# Create development environment
conda create -n safeagents python=3.10
conda activate safeagents

# Install requirements
pip install -r requirements.txt

# Install in development mode (changes reflect immediately)
pip install -e .
```

---

## Updating SafeAgents

To get the latest version:

```bash
cd SafeAgents
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

## Uninstallation

To completely remove SafeAgents:

```bash
# Remove conda environment
conda deactivate
conda env remove -n safeagents

# Remove repository
cd ..
rm -rf SafeAgents
```

---

## Next Steps

Now that SafeAgents is installed:

1. **[Quick Start →](../../QUICKSTART.md)** - Get up and running in 5 minutes
2. **[Your First Agent →](first-agent.md)** - Create your first agent
3. **[Core Concepts →](basic-concepts.md)** - Understand the framework

---

[← Back to Documentation](../README.md) | [Your First Agent →](first-agent.md)
