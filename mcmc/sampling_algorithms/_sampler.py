from abc import abstractmethod
from pathlib import Path
from typing import List

from mcmc.data_objects import Observations, Trace


class Sampler:
    @abstractmethod
    def generate_markov_chain(self, observations: List[Observations]) -> Trace:
        pass

    @abstractmethod
    def visualize_markov_chain(self, work_dir: Path) -> None:
        pass
