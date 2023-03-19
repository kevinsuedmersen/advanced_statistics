import numpy.typing as npt

from mcmc.data_objects import Dataset, Sample


class DatasetSerializer:
    @staticmethod
    def from_data(data: npt.NDArray) -> Dataset:
        samples = [Sample(value) for value in data.tolist()]
        return Dataset(samples)
