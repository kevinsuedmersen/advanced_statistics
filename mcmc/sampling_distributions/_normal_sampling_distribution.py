from typing import List, Optional

import scipy.stats as st

from mcmc.data_objects import Probability, Observations
from mcmc.data_objects import Sample
from mcmc.sampling_distributions._sampling_distribution import SamplingDistribution


class NormalSamplingDistribution(SamplingDistribution):
    """Univariate normal distribution used as a distribution from which to generate samples.

    Parameters
    ----------
    prior_mean: float
        Prior mean.
    prior_standard_deviation: float
        Prior standard deviation.
    sampling_standard_deviation: Optional[float], default=None
        If required, a separate standard deviation can be specified for generating the next sample.
        If None, the prior standard deviation will be used.

    Notes
    -----
    * Currently, only the uni-variate case is implemented. For multi-variate distributions, another
        layer of abstraction might be necessary.
    """
    def __init__(
            self,
            prior_mean: float,
            prior_standard_deviation: float,
            sampling_standard_deviation: Optional[float] = None
    ):
        self._prior_mean = prior_mean
        self._prior_standard_deviation = prior_standard_deviation
        self._sampling_standard_deviation = sampling_standard_deviation

    def generate_initial_random_sample(self) -> Sample:
        """
        Generate a first sample, usually that sample is "close" to the mean.

        Returns
        -------
        Sample
        """
        return Sample(value=self._prior_mean)

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
        sd = self._sampling_standard_deviation or self._prior_standard_deviation
        value = st.norm(sample.value, sd).rvs()
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
        likelihood = st.norm(sample.value, self._prior_standard_deviation).pdf(observations.data).prod()
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
        prior = st.norm(self._prior_mean, self._prior_standard_deviation).pdf(sample.value)
        return Probability(prior)
