import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_entropy_subplots(p_x):

    neg_log_p_x = -np.log(p_x)
    combined = p_x * neg_log_p_x

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi=300)
    suptitle = "Entropy Transformations"
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")

    sns.lineplot(ax=axes[0, 0], x=range(len(p_x)), y=p_x, color="blue", label="P(x)")
    axes[0, 0].set_title("P(x)")
    axes[0, 0].set_xlabel("Index")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(loc="upper right")

    sns.lineplot(
        ax=axes[0, 1],
        x=range(len(neg_log_p_x)),
        y=neg_log_p_x,
        color="red",
        label="- log(P(x))",
    )
    axes[0, 1].set_title("- log(P(x))")
    axes[0, 1].set_xlabel("Index")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].legend(loc="upper right")

    sns.lineplot(
        ax=axes[1, 0],
        x=range(len(combined)),
        y=combined,
        color="green",
        label="- P(x) * log(P(x))",
    )
    axes[1, 0].set_title("- P(x) * log(P(x))")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend(loc="upper right")

    sns.lineplot(
        ax=axes[1, 1],
        x=range(len(p_x)),
        y=combined,
        color="green",
        label="- P(x) * log(P(x))",
    )
    sns.lineplot(ax=axes[1, 1], x=range(len(p_x)), y=p_x, color="blue", label="P(x)")
    axes[1, 1].set_title("Combined Plot")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].legend(loc="upper right")

    # Calculate the entropy value
    entropy_value = np.sum(combined)
    fig.text(0.5, 0.03, f"Entropy: {entropy_value:.4f}", ha="center", fontsize=12)
    avg_entropy_value = entropy_value / len(combined)
    fig.text(
        0.5, 0.01, f"Average Entropy: {avg_entropy_value:.4f}", ha="center", fontsize=12
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)
    plt.show()

    return entropy_value


def plot_cross_entropy_subplots(p_x, q_x):

    neg_log_q_x = -np.log(q_x)
    combined = p_x * neg_log_q_x

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi=300)
    suptitle = "Cross Entropy Transformations"
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")

    sns.lineplot(ax=axes[0, 0], x=range(len(p_x)), y=p_x, color="blue", label="P(x)")
    axes[0, 0].set_title("P(x)")
    axes[0, 0].set_xlabel("Index")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(loc="upper right")

    ax2 = axes[0, 1].twinx()  # Create a second y-axis
    sns.lineplot(
        ax=axes[0, 1],
        x=range(len(neg_log_q_x)),
        y=neg_log_q_x,
        color="red",
        label="- log(Q(x))",
    )
    sns.lineplot(
        ax=ax2,
        x=range(len(q_x)),
        y=q_x,
        color="orange",
        label="Q(x)",
        alpha=0.5,
    )
    ax2.set_ylabel("Q(x)", color="orange")
    ax2.grid(False)
    ax2.tick_params(axis="y", labelcolor="orange")

    # Set title and labels for main axis
    axes[0, 1].set_title("- log(Q(x))")
    axes[0, 1].set_xlabel("Index")
    axes[0, 1].set_ylabel("Value")

    # Position the legends separately
    axes[0, 1].legend(loc="upper left")
    ax2.legend(loc="upper right")

    sns.lineplot(
        ax=axes[1, 0],
        x=range(len(combined)),
        y=combined,
        color="green",
        label="- P(x) * log(Q(x))",
    )
    axes[1, 0].set_title("- P(x) * log(Q(x))")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend(loc="upper right")

    sns.lineplot(
        ax=axes[1, 1],
        x=range(len(p_x)),
        y=combined,
        color="green",
        label="- P(x) * log(Q(x))",
    )
    sns.lineplot(ax=axes[1, 1], x=range(len(p_x)), y=p_x, color="blue", label="P(x)")
    axes[1, 1].set_title("Combined Plot")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].legend(loc="upper right")

    # Calculate the cross entropy value
    cross_entropy_value = np.sum(combined)
    fig.text(
        0.5, 0.03, f"Cross Entropy: {cross_entropy_value:.4f}", ha="center", fontsize=12
    )
    avg_cross_entropy_value = cross_entropy_value / len(combined)
    fig.text(
        0.5,
        0.01,
        f"Average Cross Entropy: {avg_cross_entropy_value:.4f}",
        ha="center",
        fontsize=12,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)
    plt.show()

    return cross_entropy_value


def plot_kl_divergence_subplots(p_x, q_x):
    # also D_KL(P||Q) = H(P,Q) - H(P)

    # Calculate components for KL divergence
    ratio = p_x / q_x  # P(x)/Q(x)
    log_ratio = np.log(ratio)  # log(P(x)/Q(x))
    kl_pointwise = p_x * log_ratio  # P(x) * log(P(x)/Q(x))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi=300)
    suptitle = "KL Divergence Transformations"
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")

    # Plot 1: P(x) and Q(x)
    ax1a = axes[0, 0]
    sns.lineplot(ax=ax1a, x=range(len(p_x)), y=p_x, color="blue", label="P(x)")
    ax1b = ax1a.twinx()
    sns.lineplot(
        ax=ax1b, x=range(len(q_x)), y=q_x, color="orange", label="Q(x)", alpha=0.5
    )

    ax1a.set_title("Probability Distributions")
    ax1a.set_xlabel("Index")
    ax1a.set_ylabel("P(x)", color="blue")
    ax1a.tick_params(axis="y", labelcolor="blue")
    ax1b.set_ylabel("Q(x)", color="orange")
    ax1b.tick_params(axis="y", labelcolor="orange")
    ax1b.grid(False)

    ax1a.legend(loc="upper left")
    ax1b.legend(loc="upper right")

    # Plot 2: log(P(x)/Q(x)) and P(x)/Q(x)
    ax2a = axes[0, 1]
    sns.lineplot(
        ax=ax2a,
        x=range(len(log_ratio)),
        y=log_ratio,
        color="purple",
        label="log(P(x)/Q(x))",
    )
    ax2b = ax2a.twinx()

    sns.lineplot(
        ax=ax2b,
        x=range(len(ratio)),
        y=ratio,
        color="red",
        label="P(x)/Q(x)",
        alpha=0.5,
    )

    ax2a.set_title("log(P(x)/Q(x)) and P(x)/Q(x)")
    ax2a.set_xlabel("Index")
    ax2a.set_ylabel("log(P(x)/Q(x))", color="purple")
    ax2a.tick_params(axis="y", labelcolor="purple")
    ax2b.set_ylabel("P(x)/Q(x)", color="red")
    ax2b.tick_params(axis="y", labelcolor="red")
    ax2b.grid(False)

    ax2a.legend(loc="upper left")
    ax2b.legend(loc="upper right")

    # Plot 3: P(x) * log(P(x)/Q(x))
    sns.lineplot(
        ax=axes[1, 0],
        x=range(len(kl_pointwise)),
        y=kl_pointwise,
        color="green",
        label="P(x) * log(P(x)/Q(x))",
    )
    axes[1, 0].set_title("P(x) * log(P(x)/Q(x))")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend(loc="upper right")

    # Plot 4: Combined plot
    sns.lineplot(
        ax=axes[1, 1],
        x=range(len(kl_pointwise)),
        y=kl_pointwise,
        color="green",
        label="P(x) * log(P(x)/Q(x))",
    )
    sns.lineplot(ax=axes[1, 1], x=range(len(p_x)), y=p_x, color="blue", label="P(x)")
    sns.lineplot(
        ax=axes[1, 1], x=range(len(q_x)), y=q_x, color="orange", label="Q(x)", alpha=0.5
    )
    axes[1, 1].set_title("Combined Plot")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].legend(loc="upper right")

    # Calculate the KL divergence value
    kl_value = np.sum(kl_pointwise)
    fig.text(0.5, 0.03, f"KL Divergence: {kl_value:.4f}", ha="center", fontsize=12)
    avg_kl_value = kl_value / len(kl_pointwise)
    fig.text(
        0.5,
        0.01,
        f"Average KL Divergence: {avg_kl_value:.4f}",
        ha="center",
        fontsize=12,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)
    plt.show()

    return kl_value
