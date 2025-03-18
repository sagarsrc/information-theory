# %% [markdown]
# # Information Theory
# %% [markdown]
# auto-reload modules in src folder
import notebook_setup

# %% [markdown]
# ## Imports
import numpy as np
import matplotlib.pyplot as plt

from src.distribution.gen_distribution import DistributionGenerator
from src.distribution.plot_distribution import DistributionVisualizer
from src.distribution.pdf_estimator import PDFEstimator

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

# %%
# %% [markdown]
# ## check how P(x) looks like


# %%
# Call the function to plot P_normal PDF
dv.plot_pdf(
    pdf["P_normal"],
    x_eval,
    "Probability Density Function of P_normal",
    "P_normal PDF",
    "blue",
)
# %%

dv.plot_pdf(
    pdf["Q_normal_shifted"],
    x_eval,
    "Probability Density Function of Q_normal_shifted",
    "Q_normal_shifted PDF",
    "red",
)

# %%
dv.plot_pdf(
    pdf["R_normal_wider"],
    x_eval,
    "Probability Density Function of R_normal_wider",
    "R_normal_wider PDF",
    "green",
)

# %%
dv.plot_pdf(
    pdf["S_uniform"],
    x_eval,
    "Probability Density Function of S_uniform",
    "S_uniform PDF",
    "orange",
)

# %%

dv.plot_pdf(
    pdf["T_bimodal"],
    x_eval,
    "Probability Density Function of T_bimodal",
    "T_bimodal PDF",
    "purple",
)
