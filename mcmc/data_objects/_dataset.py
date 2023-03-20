from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from beartype import beartype
from nptyping import NDArray, Shape, Float

from mcmc.custom_types import NSamples, NVars
from mcmc.data_objects._sample import Sample


@dataclass
class Dataset:
    samples: List[Sample]
    n_samples: NSamples = field(repr=True, init=False)
    n_vars: NVars = field(repr=True)

    def __post_init__(self):
        self.n_samples = NSamples(len(self.samples))

    @beartype
    @property
    def data(self) -> NDArray[Shape["NSamples, NVars"], Float]:
        # TODO (issue): Typing of return value not working: NSamples and NVars are not recognized as symbols
        values = [sample.value for sample in self.samples]
        values = np.asarray(values)
        values = values.reshape(-1, self.n_vars)  # multi-variate case is the default
        return values

    def add_sample(self, sample: Sample) -> None:
        self.samples.append(sample)

    def slice_samples(self, start_idx: int = 0, end_idx: Optional[int] = None) -> None:
        if not end_idx:
            end_idx = len(self.samples)
        self.samples = self.samples[start_idx:end_idx]
