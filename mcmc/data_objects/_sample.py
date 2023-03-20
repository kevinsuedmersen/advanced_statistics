from dataclasses import dataclass, field

from nptyping import NDArray, Shape, Float

from mcmc.custom_types import NVars


@dataclass
class Sample:
    """
    A randomly generated sample.

    Parameters
    ----------
    value: NDArray[Shape["NVars"]]
        If `value` is a float, it is one scalar coming from a uni-variate distribution.
        If `value` is a list of floats, it still represents one sample, but
        this time, it comes from a multi-variate distribution.
    """
    value: NDArray[Shape["NVars"], Float] = field(repr=True)
