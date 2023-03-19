from abc import ABC, abstractmethod
from typing import List

from mcmc.data_objects import Sample, Observations


class Sampler:
    @abstractmethod
    def generate_markov_chain(self, observations: List[Observations]) -> List[Sample]:
        pass
