from dataclasses import dataclass


@dataclass
class Probability:
    """
    A probability value.

    Parameters
    ----------
    value: float
        Probabilita value, so a float between 0 and 1.

    """
    value: float

    def __post_init__(self):
        assert 0 <= self.value <= 1, "Probability values can only be between 0 and 1."
