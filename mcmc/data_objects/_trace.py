from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from mcmc.data_objects._sample import Sample


@dataclass
class Trace:
    samples: List[Sample] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.samples, list)
        self.n_vars = self._infer_n_vars()

    @property
    def data(self) -> npt.NDArray:
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

    def _infer_n_vars(self) -> int:
        """Infers the number of variables from the first sample."""
        first_value = self.samples[0].value
        if isinstance(first_value, (float, int)):
            return 1
        elif isinstance(first_value, List):
            return len(first_value)
        else:
            raise ValueError(f"The first sample has an unexpected data structure: {first_value=}")
