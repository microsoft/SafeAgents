class ModelConfig:
    """
    The config params of Model initialization.
    """
    azure_endpoint: str = None
    azure_deployment: str = None
    azure_model_name: str = None
    azure_api_version: str = None
    azure_token_provider: str = None
    temperature: float = 0.0

class Model:
    """
    The underlying LLM to use for generating responses and actions in an Agent.
    """

    def __init__(self, config: ModelConfig, client, framework: str):
        """
        Use other class methods to create objects of this class. Avoid direct instantiation.
        """
        self.config = config
        self.client = client
        self.framework = framework

    @classmethod
    def from_azure_openai_for_autogen(cls, config: ModelConfig):

        from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
        chat_completion_kwargs = {
            "azure_endpoint": config.azure_endpoint,
            "api_version": config.azure_api_version,
            "model_capabilities": {
                "function_calling": True,
                "json_output": True,
                "vision": True,
                "structured_output": True,
            },
            "azure_ad_token_provider": config.azure_token_provider,
            "model": config.azure_model_name,
            "temperature": config.temperature,
            "azure_deployment": config.azure_deployment,
        }
        model_client = AzureOpenAIChatCompletionClient(**chat_completion_kwargs)

        framework = "Autogen"
        
        return cls(config, model_client, framework)

    def get_client(self):
        return self.client
