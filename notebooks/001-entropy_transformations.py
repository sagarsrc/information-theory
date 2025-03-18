# %% [markdown]
# # Information Theory
# %% [markdown]
# auto-reload modules in src folder
import notebook_setup

# %% [markdown]
# ## Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.distribution.gen_distribution import DistributionGenerator
from src.distribution.plot_distribution import DistributionVisualizer
from src.distribution.pdf_estimator import PDFEstimator
from src.information_theory.metrics import InformationTheory
from src.utils.entropy_plotter import (
    plot_entropy_subplots,
    plot_cross_entropy_subplots,
    plot_kl_divergence_subplots,
)

# %% [markdown]
# ## Initialize Distribution Generator and Visualizer
dg = DistributionGenerator(random_seed=1437)
dv = DistributionVisualizer(
    style="whitegrid",
    context="paper",
    palette="bright",
)

# %% [markdown]
# ## Generate Distributions and get PDFs

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
    "U_beta": dg.beta(a=1, b=2.5, size=1000),
}
pdf = {i: PDFEstimator(disb[i], method="kde") for i in disb}

# Modify the x_eval range to capture the full distributions
x_eval = np.linspace(-6, 6, 1000)  # Extended range to cover both distributions

# %% [markdown]
# ## Initialize Information Theory
it = InformationTheory()

# %% [markdown]
# ## Check entropy transformations
eps = 1e-10
p_x = pdf["P_normal"].evaluate(x_eval) + eps
q_x = pdf["R_normal_wider"].evaluate(x_eval) + eps

# %% [markdown]
# ## Plot entropy transformations
entropy_value = plot_entropy_subplots(p_x)
print(f"Entropy Value: {entropy_value:.4f}")

# %% [markdown]
# ## Plot cross entropy transformations
cross_entropy_value = plot_cross_entropy_subplots(p_x, q_x)
print(f"Cross Entropy Value: {cross_entropy_value:.4f}")

# %% [markdown]
# ## Plot KL divergence transformations
eps = 1e-10
p_x = pdf["P_normal"].evaluate(x_eval) + eps
q_x = pdf["R_normal_wider"].evaluate(x_eval) + eps
kl_value = plot_kl_divergence_subplots(p_x, q_x)
print(f"KL Divergence Value: {kl_value:.4f}")

# %%
