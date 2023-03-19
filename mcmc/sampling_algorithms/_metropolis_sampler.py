from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from mcmc.data_objects import Sample, Observations, Probability
from mcmc.sampling_algorithms._sampler import Sampler
from mcmc.sampling_distributions import SamplingDistribution


class MetropolisSampler(Sampler):
    def __init__(
            self,
            sampling_distribution: SamplingDistribution,
            total_steps: int,
            warumup_steps: int = 1_000,
            epsilon: float = 0.,  # epsilon could als be 1e-307
            verbose: bool = False
    ):
        self._sampling_distribution = sampling_distribution
        self._total_steps = warumup_steps + total_steps
        self._warumup_steps = warumup_steps
        self._epsilon = epsilon
        self._trace = []
        self._accepted_samples = 0
        self._rejected_samples = 0
        self._verbose = verbose

    @property
    def accepted_samples(self) -> int:
        return self._accepted_samples

    @property
    def rejected_samples(self) -> int:
        return self._rejected_samples

    @property
    def acceptance_ratio(self) -> float:
        return self._accepted_samples / (self._accepted_samples + self._rejected_samples)

    def generate_markov_chain(self, observations: Observations) -> List[Sample]:
        # Start the markov chain with an initial sample
        previous_sample = self._sampling_distribution.generate_initial_random_sample()
        self._trace.append(previous_sample)

        print(f"Generating a Markov Chain with {self._total_steps} samples (inclusing {self._warumup_steps} warmup steps).")
        for step in tqdm(range(self._total_steps)):
            next_sample = self._sampling_distribution.generate_next_random_sample_based_on(previous_sample)
            previous_likelihood, next_likelihood = self._compute_likelihoods(previous_sample, next_sample, observations)
            previous_prior, next_prior = self._compute_priors(previous_sample, next_sample)

            if self._accept_new_sample(previous_likelihood, previous_prior, next_likelihood, next_prior):
                self._trace.append(next_sample)
                previous_sample = next_sample
            else:
                self._trace.append(previous_sample)

            if (step % 1_000) == 0:
                print(f"Currently accepted samples: {self.accepted_samples}, rejected samples: {self.rejected_samples}, acceptance ratio: {self.acceptance_ratio}")

        # Discard the first warmup steps
        self._trace = self._trace[self._warumup_steps:]
        return self._trace

    def _compute_likelihoods(
            self,
            previous_sample: Sample,
            next_sample: Sample,
            observations: Observations
    ) -> Tuple[Probability, Probability]:
        """Compute likelihoods of previous and next sample"""
        previous_likelihood = self._sampling_distribution.compute_likelihood_based_on(
            previous_sample, observations
        )
        next_likelihood = self._sampling_distribution.compute_likelihood_based_on(
            next_sample, observations
        )
        return previous_likelihood, next_likelihood

    def _compute_priors(
            self,
            previous_sample: Sample,
            next_sample: Sample
    ) -> Tuple[Probability, Probability]:
        """Computes priors of the previous and next sample."""
        previous_prior = self._sampling_distribution.compute_prior_based_on(previous_sample)
        next_prior = self._sampling_distribution.compute_prior_based_on(next_sample)
        return previous_prior, next_prior

    def _accept_new_sample(
            self,
            previous_likelihood: Probability,
            previous_prior: Probability,
            next_likelihood: Probability,
            next_prior: Probability
    ) -> bool:
        """Apply the Metropolis-Hastings criterion to decied whether to keep the nexext sample"""
        previous_p = previous_likelihood.value * previous_prior.value
        next_p = next_likelihood.value * next_prior.value

        # Handle potential division by zero errors
        if previous_p == 0:
            if self._verbose:
                print(f"{previous_p=} is zero ==> Adding {self._epsilon} to it. At the same time, {next_p=}")
            previous_p += self._epsilon

        # Accept the next sample with a certain probability
        if np.random.randn() < (next_p / previous_p):
            self._accepted_samples += 1
            return True
        else:
            self._rejected_samples += 1
            return False
