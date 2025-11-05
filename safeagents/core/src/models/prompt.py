class Prompt:
    """
    Represents a prompt used by an agent to elicit a response or action.
    """

    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}

    def format(self, **kwargs):
        """
        Format the prompt with the given keyword arguments.
        """
        return self.content.format(**kwargs)

    def __str__(self):
        return self.content
