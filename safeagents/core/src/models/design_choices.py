from enum import Enum


class AutonomyLevel(Enum):
    HIGH = "high"
    LOW = "low"

class PlanningStrategy(Enum):
    STRATIFIED = "stratified"
    UNIFIED = "unified"

# class MessageType(Enum):
#     USER = "user"
#     SYSTEM = "system"


class DesignChoices:
    """
    Stores arbitrary design dimensions for a Team.
    Allows flexibility so that new dimensions can be added without modifying Team.
    Also validates core dimensions like autonomy_level, planning_strategy, and message_type.
    """

    ALLOWED_DIMENSIONS = {
        "autonomy_level": AutonomyLevel,
        "planning_strategy": PlanningStrategy,
        # "message_type": MessageType
    }

    def __init__(self, new_dimensions: dict[str, Enum] = {}):
        self._choices = {}
        self.add_allowed_dimensions(new_dimensions)

    def add_allowed_dimensions(self, new_dimensions: dict[str, Enum]):
        """
        Add new dimensions to the allowed dimensions.
        """
        for name, enum_cls in new_dimensions.items():
            if not isinstance(enum_cls, type) or not issubclass(enum_cls, Enum):
                raise ValueError(f"Invalid enum class for dimension '{name}'")
            self.ALLOWED_DIMENSIONS[name] = enum_cls

    def set_choices(self, **dimensions):
        """
        Initialize with key-value pairs where:
          - key: dimension name (str)
          - value: Enum member or valid string for the Enum
        """
        # Validate required dimensions first
        for name, enum_cls in self.ALLOWED_DIMENSIONS.items():
            if name not in dimensions:
                raise ValueError(f"Missing required dimension: {name}")
            value = dimensions[name]
            if isinstance(value, str):
                try:
                    value = enum_cls(value)
                except ValueError:
                    raise ValueError(f"Invalid value for '{name}': must be one of {[e.value for e in enum_cls]}")
            elif not isinstance(value, enum_cls):
                raise ValueError(f"Invalid type for '{name}': must be {enum_cls.__name__}")
            self._choices[name] = value

    def get_all_design_dimensions(self):
        """Return all design dimensions as a dictionary."""
        return list(self._choices.keys())

    def get(self, dimension_name):
        """Retrieve the value for a given dimension name, or None if not set."""
        return self._choices.get(dimension_name)

    def __getitem__(self, dimension_name):
        """Allow dict-like access to dimensions."""
        return self._choices[dimension_name]

    def __setitem__(self, dimension_name, value):
        """Allow setting dimensions after initialization."""
        if isinstance(value, Enum) or isinstance(value, str):
            self._choices[dimension_name] = value
        else:
            raise ValueError(f"Invalid value for '{dimension_name}': must be Enum")

    def __contains__(self, dimension_name):
        return dimension_name in self._choices

    def __repr__(self):
        return f"DesignChoices({self._choices})"