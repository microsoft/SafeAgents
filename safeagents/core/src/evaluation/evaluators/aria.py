"""
ARIA (Agent Risk Identification Assessment) evaluator.
"""

from ....logger import logger
from ..prompts.aria_prompt import ARIA_PROMPT


def aria(logs: str, framework: str = None) -> str:
    """
    ARIA evaluation.

    Args:
        logs: Execution logs as a string
        framework: Framework name (e.g., "autogen", "langgraph", "openai-agents").
                  Optional, can be used to adapt prompts in the future.

    Returns:
        str: ARIA score (1, 2, or 3)
    """
    from ...clients.azure_openai import get_azure_openai_client_sync, get_azure_config

    system_prompt = ARIA_PROMPT

    # Note: Framework parameter available for future use
    # Could be used to adapt prompt based on framework-specific log formats

    prompt = f""" The log content is:\n{logs}\n"""

    try:
        client = get_azure_openai_client_sync()
        config = get_azure_config()

        response = client.chat.completions.create(
            model=config["azure_deployment"],
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        resp = response.choices[0].message.content
        return resp.split("<answer>")[1].split("</answer>")[0].strip()
    except Exception as e:
        logger.error(f"ARIA evaluation failed: {e}")
        return "3"
