"""
Basic Multi-Agent System Example
=================================

This example demonstrates the fundamental building blocks of SafeAgents:
- Creating agents with custom tools
- Setting up agent handoffs
- Running a multi-agent system with different frameworks
- Using assessment metrics (ARIA & DHARMA)

Use this as a starting point for building your own multi-agent applications.
"""

import asyncio
import os
from dotenv import load_dotenv

from safeagents import Agent, AgentConfig, Team, tool

# Load environment variables
load_dotenv()


# ============================================================================
# Step 1: Define Custom Tools
# ============================================================================

@tool()
def get_weather(city: str) -> str:
    """
    Get weather information for a city.

    Args:
        city: Name of the city

    Returns:
        Weather information as a string
    """
    # In a real application, this would call a weather API
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C",
        "Tokyo": "Rainy, 20°C"
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@tool()
async def get_traffic(city: str) -> str:
    """
    Get traffic information for a city.

    Args:
        city: Name of the city

    Returns:
        Traffic information as a string
    """
    # Async tools are also supported
    traffic_data = {
        "New York": "Heavy traffic on I-95",
        "London": "Moderate traffic on M25",
        "Tokyo": "Light traffic on Route 1"
    }
    return traffic_data.get(city, f"Traffic data not available for {city}")


# ============================================================================
# Step 2: Create Agents with Tools
# ============================================================================

def create_agents():
    """Create weather and traffic agents."""

    # Weather Agent - specializes in weather information
    weather_agent = Agent(
        config=AgentConfig(
            name="WeatherAgent",
            tools=[get_weather],
            system_message=(
                "You are a helpful weather assistant. "
                "Use your weather tool to provide accurate weather information. "
                "If asked about traffic, transfer to the TrafficAgent."
            ),
            description="An agent that provides weather information",
            handoff_description=(
                "Transfer to WeatherAgent for weather-related questions"
            ),
            handoffs=["TrafficAgent"]  # Can transfer to TrafficAgent
        )
    )

    # Traffic Agent - specializes in traffic information
    traffic_agent = Agent(
        config=AgentConfig(
            name="TrafficAgent",
            tools=[get_traffic],
            system_message=(
                "You are a helpful traffic assistant. "
                "Use your traffic tool to provide accurate traffic information. "
                "If asked about weather, transfer to the WeatherAgent."
            ),
            description="An agent that provides traffic information",
            handoff_description=(
                "Transfer to TrafficAgent for traffic-related questions"
            ),
            handoffs=["WeatherAgent"]  # Can transfer back to WeatherAgent
        )
    )

    return [weather_agent, traffic_agent]


# ============================================================================
# Step 3: Run with Different Frameworks
# ============================================================================

async def run_example(framework="openai-agents", architecture="centralized"):
    """
    Run the multi-agent system.

    Args:
        framework: Framework to use ("openai-agents", "autogen", or "langgraph")
        architecture: Architecture type ("centralized" or "decentralized")
    """
    print(f"\n{'='*80}")
    print(f"Running with {framework} - {architecture}")
    print(f"{'='*80}\n")

    # Create agents
    agents = create_agents()

    # Create team
    team = Team.create(
        agents=agents,
        framework=framework,
        architecture=architecture,
        max_turns=10
    )

    # Run task with assessment
    task = "What is the weather and traffic like in New York?"

    print(f"Task: {task}\n")

    result = await team.run(
        task=task,
        verbose=True,  # Print execution logs
        assessment=["aria", "dharma"]  # Enable safety assessment
    )

    # Display results
    print(f"\n{'='*80}")
    print("Results")
    print(f"{'='*80}")
    print(f"Stop Reason: {result.get('stop_reason', 'N/A')}")
    print(f"Messages: {len(result.get('messages', []))} messages exchanged")
    print(f"\nAssessment:")
    print(f"  ARIA Level: {result.get('assessment', {}).get('aria', 'N/A')}")
    print(f"  DHARMA Classification: {result.get('assessment', {}).get('dharma', 'N/A')}")
    print(f"\nExecution Trace:")
    print(f"  Tools Called: {len(result.get('execution_trace', {}).get('tool_calls', []))}")

    return result


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point with CLI argument support."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Basic Multi-Agent System Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with OpenAI Agents (default)
  python 01_basic_multi_agent.py

  # Run with Autogen
  python 01_basic_multi_agent.py --framework autogen

  # Run with LangGraph in decentralized mode
  python 01_basic_multi_agent.py --framework langgraph --architecture decentralized
        """
    )

    parser.add_argument(
        "--framework",
        type=str,
        default="openai-agents",
        choices=["openai-agents", "autogen", "langgraph"],
        help="Framework to use (default: openai-agents)"
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="centralized",
        choices=["centralized", "decentralized"],
        help="Architecture type (default: centralized)"
    )

    args = parser.parse_args()

    # Run the example
    asyncio.run(run_example(
        framework=args.framework,
        architecture=args.architecture
    ))


if __name__ == "__main__":
    main()
