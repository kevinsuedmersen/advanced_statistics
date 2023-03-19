import numpy.typing as npt

from dataclasses import dataclass, field


@dataclass
class Observations:
    data: npt.NDArray
