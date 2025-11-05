import os
from dotenv import load_dotenv
from azure.identity import AzureCliCredential
from azure.identity.aio import get_bearer_token_provider

class EnvironmentSetup:
    """
    Encapsulates the configuration and initialization of the environment in which the multi-agent system operates.
    """

    def __init__(self):
        load_dotenv()
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_DEPLOYMENT")
        self.azure_model_name = os.getenv("AZURE_MODEL_NAME")
        self.azure_api_version = os.getenv("AZURE_API_VERSION")
        self.framework = os.getenv("FRAMEWORK")
        self.exp_type = os.getenv("EXP_TYPE")
        self.credential = AzureCliCredential()
        self.token_provider = get_bearer_token_provider(self.credential, "api://trapi/.default")

    def get_framework(self):
        return self.framework

    def get_exp_type(self):
        return self.exp_type

    def get_azure_config(self):
        return {
            "endpoint": self.azure_endpoint,
            "deployment": self.azure_deployment,
            "model_name": self.azure_model_name,
            "api_version": self.azure_api_version,
            "token_provider": self.token_provider
        }
