"""
Hello World - SafeAgents Quick Start
=====================================

The simplest possible SafeAgents example.
Perfect for getting started and verifying your installation.

This example:
1. Creates a simple agent with one tool
2. Runs a basic task
3. Shows the result

Run this first to make sure everything is working!
"""

import asyncio
from dotenv import load_dotenv

from safeagents import Agent, AgentConfig, Team, tool

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Step 1: Define a Simple Tool
# ============================================================================

@tool()
def greet(name: str) -> str:
    """
    Greet someone by name.

    Args:
        name: The name of the person to greet

    Returns:
        A friendly greeting
    """
    return f"Hello, {name}! Welcome to SafeAgents!"


# ============================================================================
# Step 2: Create an Agent
# ============================================================================

def create_greeter_agent():
    """Create a simple greeting agent."""
    agent = Agent(
        config=AgentConfig(
            name="GreeterAgent",
            tools=[greet],
            system_message="You are a friendly assistant that greets people."
        )
    )
    return agent


# ============================================================================
# Step 3: Run a Task
# ============================================================================

async def main():
    """Main function to run the example."""

    print("\n" + "="*80)
    print("SafeAgents - Hello World Example")
    print("="*80 + "\n")

    # Create agent
    agent = create_greeter_agent()

    # Create team
    team = Team.create(
        agents=[agent],
        framework="openai-agents",  # or "autogen", "langgraph"
        architecture="centralized",
        max_turns=3
    )

    # Run task
    task = "Please greet Alice"
    print(f"Task: {task}\n")

    result = await team.run(
        task=task,
        verbose=True  # Show execution details
    )

    # Show result
    print("\n" + "="*80)
    print("Result")
    print("="*80)
    print(f"Status: {result.get('stop_reason', 'completed')}")
    print(f"Messages: {len(result.get('messages', []))} exchanged")

    # Get the final message
    if result.get('messages'):
        final_message = result['messages'][-1]
        print(f"\nFinal response:\n{final_message.get('content', 'N/A')}")

    print("\n✅ Success! SafeAgents is working correctly.\n")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())
