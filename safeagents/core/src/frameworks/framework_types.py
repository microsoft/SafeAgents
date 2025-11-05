from enum import Enum


class Framework(Enum):
    AUTOGEN = "autogen"
    LANGGRAPH = "langgraph"
    OPENAI_AGENTS = "openai-agents"
    # Future frameworks can be easily added here
    # CREWAI = "crewai"
    # SWARM = "swarm"