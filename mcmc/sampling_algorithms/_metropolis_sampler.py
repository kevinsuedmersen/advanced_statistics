from typing import List, Tuple

import numpy as np

from mcmc.data_objects import Sample, Observations, Probability
from mcmc.sampling_algorithms._sampler import Sampler
from mcmc.sampling_distributions import SamplingDistribution


class MetropolisSampler(Sampler):
    def __init__(self, sampling_distribution: SamplingDistribution, steps: int):
        self._sampling_distribution = sampling_distribution
        self._steps = steps
        self._trace = []

    def generate_markov_chain(self, observations: Observations) -> List[Sample]:
        # Start the markov chain with an initial sample
        previous_sample = self._sampling_distribution.generate_initial_random_sample()
        self._trace.append(previous_sample)

        for step in range(self._steps):
            next_sample = self._sampling_distribution.generate_next_random_sample_based_on(previous_sample)
            previous_likelihood, next_likelihood = self._compute_likelihoods(previous_sample, next_sample, observations)
            previous_prior, next_prior = self._compute_priors(previous_sample, next_sample)

            if self._accept_new_sample(previous_likelihood, previous_prior, next_likelihood, next_prior):
                self._trace.append(next_sample)
                previous_sample = next_sample
            else:
                self._trace.append(previous_sample)

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

    @staticmethod
    def _accept_new_sample(
            previous_likelihood: Probability,
            previous_prior: Probability,
            next_likelihood: Probability,
            next_prior: Probability
    ) -> bool:
        """Apply the Metropolis-Hastings criterion to decied whether to keep the nexext sample"""
        previous_p = previous_likelihood.value * previous_prior.value
        next_p = next_likelihood.value * next_prior.value
        if np.random.randn() < (next_p / previous_p):
            return True
        else:
            return False