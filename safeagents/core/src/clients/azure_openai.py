"""
Shared Azure OpenAI client utilities for SafeAgents.
Ensures consistent client creation across frameworks and assessment.
"""
import os
from typing import Dict, Any
from azure.identity import AzureCliCredential
from azure.identity.aio import get_bearer_token_provider


def get_azure_config() -> Dict[str, Any]:
    """
    Get Azure OpenAI configuration from environment variables.
    Used by both frameworks and assessment modules.

    Returns:
        dict: Configuration dictionary with Azure OpenAI settings

    Raises:
        ValueError: If required environment variables are not set
    """
    azure_openai_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_openai_deployment = os.getenv("AZURE_DEPLOYMENT")
    azure_openai_api_version = os.getenv("AZURE_API_VERSION")
    azure_model_name = os.getenv("AZURE_MODEL_NAME")
    azure_scope = os.getenv("AZURE_SCOPE")
    azure_temperature = float(os.getenv("AZURE_TEMPERATURE", "0.0"))

    required_vars = [
        azure_openai_endpoint,
        azure_openai_deployment,
        azure_openai_api_version,
        azure_model_name,
        azure_scope
    ]
    if any(v is None or v == "" for v in required_vars):
        raise ValueError(
            "Azure OpenAI environment variables: ("
            "AZURE_ENDPOINT, "
            "AZURE_DEPLOYMENT, "
            "AZURE_API_VERSION, "
            "AZURE_MODEL_NAME, "
            "AZURE_SCOPE, "
            "AZURE_TEMPERATURE"
            ") are not set properly in the `.env`."
        )

    credential = AzureCliCredential()
    get_bearer_token_provider_func = get_bearer_token_provider(
        credential, azure_scope
    )

    config = {
        "azure_endpoint": azure_openai_endpoint,
        "api_version": azure_openai_api_version,
        "model_capabilities": {
            "function_calling": True,
            "json_output": True,
            "vision": True,
            "structured_output": True,
        },
        "azure_ad_token_provider": get_bearer_token_provider_func,
        "model": azure_model_name,
        "temperature": azure_temperature,
        "azure_deployment": azure_openai_deployment,
    }
    return config


def get_azure_openai_client_sync():
    """
    Create a synchronous Azure OpenAI client for assessment/evaluation.
    Uses the same configuration as frameworks.

    Returns:
        AzureOpenAI: Synchronous Azure OpenAI client
    """
    from openai import AzureOpenAI
    from azure.identity import AzureCliCredential, get_bearer_token_provider

    config = get_azure_config()

    # For sync client, we need a sync token provider
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(
        credential,
        os.getenv("AZURE_SCOPE", "https://cognitiveservices.azure.com/.default")
    )

    client = AzureOpenAI(
        azure_endpoint=config["azure_endpoint"],
        api_version=config["api_version"],
        azure_ad_token_provider=token_provider,
    )

    return client


def get_azure_openai_client_async():
    """
    Create an async Azure OpenAI client.
    Uses the same configuration as frameworks.

    Returns:
        AsyncAzureOpenAI: Async Azure OpenAI client
    """
    from openai import AsyncAzureOpenAI

    config = get_azure_config()

    client = AsyncAzureOpenAI(
        azure_endpoint=config["azure_endpoint"],
        api_version=config["api_version"],
        azure_ad_token_provider=config["azure_ad_token_provider"],
    )

    return client
