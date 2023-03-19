"""Kontrollaufgabe 3.5"""
import scipy.stats as st

from mcmc.data_objects import Observations
from mcmc.sampling_algorithms import MetropolisSampler
from mcmc.sampling_distributions import NormalSamplingDistribution

if __name__ == '__main__':
    # Assume we made the following observations
    data = st.norm(450, 50).rvs(25)
    observations = Observations(data)

    STEPS = 5_000
    PRIOR_MEAN = 500
    PRIOR_STANDARD_DEVIATION = 70
    SAMPLING_STANDARD_DEVIATION = 100

    sampling_distribution = NormalSamplingDistribution(PRIOR_MEAN, PRIOR_STANDARD_DEVIATION, SAMPLING_STANDARD_DEVIATION)
    metropolis_sampler = MetropolisSampler(sampling_distribution, STEPS)
    trace = metropolis_sampler.generate_markov_chain(observations)
    print(f"{metropolis_sampler.accepted_samples=}")
    print(f"{metropolis_sampler.rejected_samples=}")
    print(f"{metropolis_sampler.acceptance_ratio=}")
