from abc import ABC, abstractmethod


class Placeholder(ABC):
    allowed_expressions = None

    @abstractmethod
    def get_expression(self, *args, **kwargs):
        """
        This function is used to get placeholders' expressions.
        It differs in different types of placeholder items.
        E.g., for placeholder object (like a "cube", a "dax", etc.),
        possible expressions include image, name, novel name, aliases, etc.
        For placeholder verb (like "shake" in novel concept grounding tasks),
        possible expressions may include name and demos showing milestones.
        """
        return
