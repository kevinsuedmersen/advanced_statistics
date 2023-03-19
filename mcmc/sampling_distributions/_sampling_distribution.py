from abc import ABC, abstractmethod

from mcmc.data_objects import Probability, Dataset
from mcmc.data_objects import Sample


class SamplingDistribution(ABC):
    """Sampling distribution interface."""
    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def generate_initial_random_sample(self) -> Sample:
        pass

    @abstractmethod
    def generate_next_random_sample_based_on(self, sample: Sample) -> Sample:
        pass

    @abstractmethod
    def compute_likelihood_based_on(self, sample: Sample, observations: Dataset) -> Probability:
        pass

    @abstractmethod
    def compute_prior_based_on(self, sample: Sample) -> Probability:
        pass
