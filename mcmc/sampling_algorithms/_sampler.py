from abc import abstractmethod
from pathlib import Path
from typing import List

from mcmc.data_objects import Dataset


class Sampler:
    @abstractmethod
    def generate_markov_chain(self, observations: List[Dataset]) -> Dataset:
        pass

    @abstractmethod
    def visualize_markov_chain(self, work_dir: Path) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
