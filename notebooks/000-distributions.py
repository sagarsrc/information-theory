# %% [markdown]
# # Information Theory
# %% [markdown]
# auto-reload modules in src folder
import notebook_setup

# %% [markdown]
# ## Imports
import numpy as np


from src.distribution.gen_distribution import DistributionGenerator
from src.distribution.plot_distribution import DistributionVisualizer
from src.distribution.pdf_estimator import PDFEstimator
from src.information_theory.metrics import InformationTheory

# %% [markdown]
# ## Initialize Distribution Generator and Visualizer
dg = DistributionGenerator(random_seed=1437)
dv = DistributionVisualizer(
    style="whitegrid",
    context="paper",
    palette="bright",
)

# %% [markdown]
# ## Generate Distributions with Clear Differences

disb = {
    "P_normal": dg.normal(mean=0, std=1, size=1000),
    "Q_normal_shifted": dg.normal(mean=2, std=1, size=1000),  # Shifted normal
    "R_normal_wider": dg.normal(mean=0, std=2, size=1000),  # Wider normal
    "S_uniform": dg.uniform(low=-3, high=3, size=1000),  # Uniform
    "T_bimodal": np.concatenate(
        [  # Bimodal
            dg.normal(mean=-1.5, std=0.5, size=500),
            dg.normal(mean=1.5, std=0.5, size=500),
        ]
    ),
}

# Common evaluation points for consistent comparison
x_eval = np.linspace(-6, 6, 1000)

# %% [markdown]
# ## Probability Density Function (PDF) Estimation
# we now estimate the probability density function of P and Q

pdf = {i: PDFEstimator(disb[i], method="kde") for i in disb}

# %% [markdown]
# ## Visualize Distributions
# %% [markdown]
# let's take a look at distribution of X
_ = dv.plot_single_distribution(disb["P_normal"], stat="density")

# %% [markdown]
# let's take a look at distribution of P and Q
_ = dv.plot_distributions_grid({k: disb[k] for k in disb}, stat="density")

_ = dv.plot_multiple_distributions(
    {k: disb[k] for k in disb if k != "X"}, stat="density"
)

_ = dv.plot_violin_box({k: disb[k] for k in disb if k != "X"}, stat="density")
