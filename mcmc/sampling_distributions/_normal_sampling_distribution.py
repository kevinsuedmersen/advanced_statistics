from typing import List

import scipy.stats as st

from mcmc.data_objects import Probability, Observations
from mcmc.data_objects import Sample
from mcmc.sampling_distributions._sampling_distribution import SamplingDistribution


class UnivariateNormalSamplingDistribution(SamplingDistribution):
    def __init__(self, mean: float, standard_deviation: float):
        self._mean = mean
        self._standard_deviation = standard_deviation

    def generate_initial_random_sample(self) -> Sample:
        """
        Generate a first sample, usually that sample is "close" to the mean.

        Returns
        -------
        Sample
        """
        return Sample(value=self._mean)

    def generate_next_random_sample_based_on(self, sample: Sample) -> Sample:
        """
        Generate a new sample with a normal distribution centered around the given sample.

        Parameters
        ----------
        sample: Sample
            The sample to center the normal distribution around.

        Returns
        -------
        Sample
        """
        value = st.norm(sample.value, self._standard_deviation).rvs()
        return Sample(value)

    def compute_likelihood_based_on(self, sample: Sample, observations: Observations) -> Probability:
        """
        Compute the likelihoods, i.e. the probabilites of observing the data with a normal
        distribution centered around the given sample.

        Parameters
        ----------
        sample: Sample
            The sample to center the normal distribution around.
        observations: Observations
            Observed data.

        Returns
        -------
        Probability
        """
        likelihood = st.norm(sample.value, self._standard_deviation).pdf(observations.data).prod()
        return Probability(likelihood)

    def compute_prior_based_on(self, sample: Sample) -> Probability:
        """
        Compute the priors, i.e. the densities of the given sample.

        Parameters
        ----------
        sample: Sample
            The sample to compute the prior for.

        Returns
        -------
        Probability
        """
        prior = st.norm(self._mean, self._standard_deviation).pdf(sample.value)
        return Probability(prior)
