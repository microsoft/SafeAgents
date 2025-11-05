class Mitigation:
    """
    Strategies and techniques used to reduce potential risks and improve the safety and reliability
    of the multi-agent system. Could be prompt-based mitigations.
    """

    def __init__(self, strategy_name, description=None, apply_fn=None):
        self.strategy_name = strategy_name
        self.description = description
        self.apply_fn = apply_fn  # Callable to apply the mitigation

    def apply(self, task):
        """
        Apply the mitigation strategy to the given task.
        """
        if self.apply_fn:
            return self.apply_fn(task)
        return task
