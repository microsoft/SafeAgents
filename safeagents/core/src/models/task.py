import json

class Task:
    """
    Defines a specific task or challenge that the multi-agent system needs to solve.
    """

    def __init__(self, prompt, category=None, target_functions=None, metadata=None):
        self.prompt = prompt
        self.category = category
        self.target_functions = target_functions or []
        self.metadata = metadata or {}

    @classmethod
    def from_json(cls, json_data):
        """
        Create a Task instance from a JSON object or file path.
        """
        if isinstance(json_data, str):
            with open(json_data, "r") as f:
                data = json.load(f)
        else:
            data = json_data

        return cls(
            prompt=data.get("prompt"),
            category=data.get("category"),
            target_functions=data.get("target_functions", []),
            metadata={k: v for k, v in data.items() if k not in ["prompt", "category", "target_functions"]},
        )

    def __str__(self):
        return f"Task(category={self.category}, prompt={self.prompt}, target_functions={self.target_functions})"
