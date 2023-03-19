"""Kontrollaufgabe 3.5"""
from datetime import datetime
from pathlib import Path

import scipy.stats as st

from mcmc.data_object_serializers import DatasetSerializer
from mcmc.sampling_algorithms import MetropolisSampler
from mcmc.sampling_distributions import NormalSamplingDistribution

if __name__ == '__main__':
    # Assume we made the following observations
    data = st.norm(450, 50).rvs(25)
    observations = DatasetSerializer.from_data(data)

    # Config
    WORK_DIR = Path("results")
    RESULTS_DIR = WORK_DIR / Path(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
    STEPS = 5_000
    PRIOR_MEAN = 500
    PRIOR_STANDARD_DEVIATION = 70
    SAMPLING_STANDARD_DEVIATION = 25

    # Start the sampling process
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir()
    sampling_distribution = NormalSamplingDistribution(
        PRIOR_MEAN, PRIOR_STANDARD_DEVIATION, SAMPLING_STANDARD_DEVIATION
    )
    metropolis_sampler = MetropolisSampler(sampling_distribution, STEPS)
    trace = metropolis_sampler.generate_markov_chain(observations)
    print(f"{metropolis_sampler.accepted_samples=}")
    print(f"{metropolis_sampler.rejected_samples=}")
    print(f"{metropolis_sampler.acceptance_ratio=}")
    metropolis_sampler.visualize_markov_chain(RESULTS_DIR)
