"""
Assessment class for evaluating multi-agent system logs.
"""

from ...logger import logger
from .evaluators.aria import aria
from .evaluators.dharma import dharma


class Assessment:
    """
    Represents the evaluation to perform: ARIA/DHARMA or other.
    """

    # Registry of available assessment methods
    METHODS = {
        "aria": aria,
        "dharma": dharma
    }

    def __init__(self, method_name, description=None, eval_fn=None):
        self.method_name = method_name
        self.description = description
        self.eval_fn = eval_fn  # Callable to perform the assessment

    def evaluate(self, team, task):
        """
        Evaluate the performance of the team on the given task.
        """
        if self.eval_fn:
            return self.eval_fn(team, task)
        return None

    @classmethod
    def evaluate_logs(cls, logs: str, methods: list[str] = None, framework: str = None) -> dict:
        """
        Evaluate logs using specified assessment methods.

        Args:
            logs: The execution logs as a string
            methods: List of method names to run (e.g., ["aria", "dharma"]).
                     If None, runs all available methods.
            framework: Framework name (e.g., "autogen", "langgraph", "openai-agents").
                      Used by evaluators to adapt prompts/logic.

        Returns:
            dict: Assessment results {method_name: score}
        """
        if methods is None:
            methods = list(cls.METHODS.keys())

        results = {}
        for method in methods:
            if method not in cls.METHODS:
                logger.warning(f"Unknown assessment method '{method}'. Skipping.")
                continue

            try:
                # Pass framework to evaluators that support it
                eval_fn = cls.METHODS[method]
                # Check if the evaluator accepts framework parameter
                import inspect
                sig = inspect.signature(eval_fn)
                if 'framework' in sig.parameters:
                    score = eval_fn(logs, framework=framework)
                else:
                    score = eval_fn(logs)
                results[method] = score
            except Exception as e:
                logger.error(f"Assessment method '{method}' failed: {e}")
                results[method] = None

        return results

    @classmethod
    def register_method(cls, name: str, eval_fn):
        """
        Register a custom assessment method.

        Args:
            name: Name of the assessment method
            eval_fn: Function that takes logs (str) and returns a score
        """
        cls.METHODS[name] = eval_fn
