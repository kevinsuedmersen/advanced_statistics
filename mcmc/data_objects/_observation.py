import numpy.typing as npt

from dataclasses import dataclass


@dataclass
class Observations:
    data: npt.NDArray
