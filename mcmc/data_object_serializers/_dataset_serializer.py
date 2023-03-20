import numpy.typing as npt

from mcmc.data_objects import Dataset, Sample


class DatasetSerializer:
    @staticmethod
    def from_data(data: npt.NDArray) -> Dataset:
        samples = [Sample(value) for value in data]
        n_vars = data.shape[1]
        return Dataset(samples, n_vars)

    @staticmethod
    def from_sample(sample: Sample) -> Dataset:
        n_vars = sample.value.shape[0]
        return Dataset([sample], n_vars)
