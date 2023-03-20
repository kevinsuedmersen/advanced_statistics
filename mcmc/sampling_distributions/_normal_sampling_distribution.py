from typing import Optional

import scipy.stats as st
from beartype import beartype
from nptyping import NDArray, Float, Shape

from mcmc.custom_types import NVars
from mcmc.data_objects import Probability, Dataset
from mcmc.data_objects import Sample
from mcmc.sampling_distributions._sampling_distribution import SamplingDistribution


class NormalSamplingDistribution(SamplingDistribution):
    """Multivariate normal distribution used as a distribution from which to generate samples.

    Parameters
    ----------
    prior_means: float
        Prior mean.
    prior_covariances: float
        Prior standard deviation.
    sampling_covariances: Optional[float], default=None
        If required, a separate standard deviation can be specified for generating the next sample.
        If None, the prior standard deviation will be used.

    Notes
    -----
    * Currently, only the uni-variate case is implemented. For multi-variate distributions, another
        layer of abstraction might be necessary.
    """
    @beartype
    def __init__(
            self,
            prior_means: NDArray[Shape["NVars"], Float],
            prior_covariances: NDArray[Shape["NVars, NVars"], Float],
            sampling_covariances: Optional[NDArray[Shape["NVars, NVars"], Float]] = None
    ):
        self._prior_means = prior_means
        self._prior_covariances = prior_covariances
        self._sampling_covariances = sampling_covariances

    def __repr__(self) -> str:
        string_repr = (
            "NormalSamplingDistribution("
            f"prior_mean={self._prior_means}, "
            f"prior_standard_deviation={self._prior_covariances}, "
            f"sampling_standard_deviation={self._sampling_covariances}"
            ")"
        )
        return string_repr

    def generate_initial_random_sample(self) -> Sample:
        """
        Generate a first sample, usually that sample is "close" to the mean.

        Returns
        -------
        Sample
        """
        return Sample(value=self._prior_means)

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
        cov = self._sampling_covariances or self._prior_covariances
        value = st.multivariate_normal(sample.value, cov).rvs()
        return Sample(value)

    def compute_likelihood_based_on(self, sample: Sample, observations: Dataset) -> Probability:
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
        likelihood = st.multivariate_normal(sample.value, self._prior_covariances).pdf(observations.data).prod()
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
        prior = st.multivariate_normal(self._prior_means, self._prior_covariances).pdf(sample.value)
        return Probability(prior)
