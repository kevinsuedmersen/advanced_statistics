from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class Sample:
    """
    A randomly generated sample.

    Parameters
    ----------
    value: float | List[float]
        If `value` is a float, it is one scalar coming from a uni-variate distribution.
        If `value` is a list of floats, it still represents one sample, but
        this time, it comes from a multi-variate distribution.
    """
    value: Union[float, List[float]] = field(repr=True, default_factory=list)
