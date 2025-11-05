from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.base import Handoff as HandoffBase
from typing import List


class CoderAgent(MagenticOneCoderAgent):

    def __init__(self, *args, handoffs: List[HandoffBase | str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        if handoffs is not None:
            self._system_messages[0] = self._system_messages[0] + "\nHANDOFF once you have the answer to the query."

        
        