from .base import Placeholder


class PlaceholderText(Placeholder):
    def __init__(
        self,
        text: str,
    ):
        self.text = text

    def get_expression(self, *args, **kwargs):
        return dict(text=self.text)
