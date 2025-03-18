import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Union, Callable, Tuple


class DistributionVisualizer:
    """
    A class for visualizing probability distributions with enhanced aesthetics using Seaborn.
    """

    def __init__(
        self,
        style="darkgrid",
        context="notebook",
        palette="deep",
        figsize=(10, 6),
        dpi=300,
    ):
        """
        Initialize the distribution visualizer with Seaborn styling.

        Parameters:
        -----------
        style : str
            Seaborn style theme ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
        context : str
            Seaborn context ('paper', 'notebook', 'talk', 'poster')
        palette : str or list
            Seaborn color palette name or custom palette
        figsize : tuple
            Default figure size (width, height) in inches
        dpi : int
            Resolution in dots per inch for the figure
        """
        # Set Seaborn aesthetics
        sns.set_theme(style=style, context=context, palette=palette)
        self.figsize = figsize
        self.dpi = dpi
        self.current_palette = sns.color_palette(palette)

    def plot_single_distribution(
        self,
        data,
        title="Distribution",
        bins=30,
        kde=True,
        hist=True,
        rug=True,
        color=None,
        fill=True,
        ax=None,
        alpha=0.6,
        stat="density",
        dpi=None,
        **kwargs
    ):
        """
        Visualize a single distribution with enhanced aesthetics.

        Parameters:
        -----------
        data : array-like
            Data samples to visualize
        title : str
            Title of the plot
        bins : int or array-like
            Number of bins or bin edges for histogram
        kde : bool
            Whether to plot kernel density estimate
        hist : bool
            Whether to plot histogram
        rug : bool
            Whether to draw a rugplot on the x-axis
        color : str or tuple
            Color for the plot elements
        fill : bool
            Whether to fill the KDE curve
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes to plot on
        alpha : float
            Transparency level (0-1)
        stat : str
            Statistic to compute for histogram ('count', 'frequency', 'density', 'probability')
        dpi : int, optional
            Resolution in dots per inch for the figure. If None, uses the class default.
        **kwargs :
            Additional keyword arguments for sns.histplot

        Returns:
        --------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        # Create figure if ax not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=dpi or self.dpi)

        # Plot histogram with KDE
        if hist:
            sns.histplot(
                data,
                bins=bins,
                kde=kde,
                stat=stat,
                alpha=alpha,
                color=color,
                ax=ax,
                **kwargs
            )
        # Plot only KDE if hist is False
        elif kde:
            sns.kdeplot(data, fill=fill, alpha=alpha, color=color, ax=ax, **kwargs)

        # Add rug plot if requested
        if rug:
            sns.rugplot(data, color=color, alpha=max(0.3, alpha - 0.3), ax=ax)

        # Customize plot appearance
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel(
            "Density" if stat == "density" else stat.capitalize(), fontsize=12
        )

        # Add grid with low alpha
        ax.grid(alpha=0.2)
        plt.tight_layout()

        return ax

    def plot_multiple_distributions(
        self,
        data_dict,
        title="Distribution Comparison",
        kde_only=False,
        hist=True,
        kde=True,
        alpha=0.5,
        bins=30,
        figsize=None,
        dpi=None,
        legend_title=None,
        colors=None,
        **kwargs
    ):
        """
        Visualize multiple distributions overlapping each other in a single plot.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping distribution names to sample arrays
        title : str
            Title of the plot
        kde_only : bool
            If True, only shows KDE curves without histograms
        hist : bool
            Whether to plot histograms
        kde : bool
            Whether to plot kernel density estimates
        alpha : float
            Transparency level (0-1)
        bins : int
            Number of bins for histograms
        figsize : tuple, optional
            Figure size (width, height) in inches
        dpi : int, optional
            Resolution in dots per inch for the figure. If None, uses the class default.
        legend_title : str, optional
            Title for the legend
        colors : list, optional
            List of colors to use for the distributions
        **kwargs :
            Additional keyword arguments for sns.kdeplot or sns.histplot

        Returns:
        --------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        if figsize is None:
            figsize = self.figsize

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi or self.dpi)

        # Use specified colors or get from current palette
        if colors is None:
            colors = self.current_palette

        # Ensure we have enough colors
        n_colors = len(data_dict)
        if len(colors) < n_colors:
            colors = sns.color_palette(n_colors=n_colors)

        # Extract stat parameter from kwargs if it exists
        stat = kwargs.pop("stat", "density")

        # Plot each distribution
        for i, (name, data) in enumerate(data_dict.items()):
            color = colors[i % len(colors)]

            if kde_only:
                sns.kdeplot(
                    data,
                    label=name,
                    fill=True,
                    alpha=alpha,
                    color=color,
                    ax=ax,
                    **kwargs
                )
            else:
                if hist:
                    sns.histplot(
                        data,
                        label=name,
                        kde=kde,
                        alpha=alpha,
                        bins=bins,
                        stat=stat,  # Use the extracted stat parameter
                        color=color,
                        ax=ax,
                        **kwargs
                    )
                elif kde:
                    sns.kdeplot(
                        data,
                        label=name,
                        fill=True,
                        alpha=alpha,
                        color=color,
                        ax=ax,
                        **kwargs
                    )

        # Customize plot appearance
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Value", fontsize=12)

        # Set appropriate y-label based on stat
        if stat == "density":
            y_label = "Density"
        elif stat == "frequency":
            y_label = "Frequency"
        elif stat == "count":
            y_label = "Count"
        elif stat == "probability":
            y_label = "Probability"
        else:
            y_label = stat.capitalize()

        ax.set_ylabel(y_label, fontsize=12)

        # Customize legend
        if legend_title:
            ax.legend(title=legend_title, fontsize=10, title_fontsize=12)
        else:
            ax.legend(fontsize=10)

        # Add grid with low alpha
        ax.grid(alpha=0.2)
        plt.tight_layout()

        return ax

    def plot_distributions_grid(
        self,
        data_dict,
        ncols=2,
        kde_only=False,
        hist=True,
        kde=True,
        rug=False,
        bins=30,
        figsize=None,
        dpi=None,
        sharex=False,
        sharey=False,
        colors=None,
        alpha=0.6,
        **kwargs
    ):
        """
        Visualize multiple distributions in a grid of subplots.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping distribution names to sample arrays
        ncols : int
            Number of columns in the grid
        kde_only : bool
            If True, only shows KDE curves without histograms
        hist : bool
            Whether to plot histograms
        kde : bool
            Whether to plot kernel density estimates
        rug : bool
            Whether to add rug plots to each distribution
        bins : int
            Number of bins for histograms
        figsize : tuple, optional
            Figure size (width, height) in inches
        dpi : int, optional
            Resolution in dots per inch for the figure. If None, uses the class default.
        sharex, sharey : bool
            Whether to share x and y axes across subplots
        colors : list, optional
            List of colors to use for the distributions
        alpha : float
            Transparency level (0-1)
        **kwargs :
            Additional keyword arguments for sns.kdeplot or sns.histplot

        Returns:
        --------
        tuple
            Figure and axes objects containing the plots
        """
        n_plots = len(data_dict)
        nrows = (n_plots + ncols - 1) // ncols  # Ceiling division

        if figsize is None:
            # Calculate appropriate figure size based on number of plots
            figsize = (5 * ncols, 4 * nrows)

        # Create subplot grid
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            dpi=dpi or self.dpi,
            sharex=sharex,
            sharey=sharey,
        )

        # Flatten axes array for easy indexing
        if n_plots > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        # Use specified colors or get from current palette
        if colors is None:
            colors = sns.color_palette(n_colors=n_plots)

        # Extract stat parameter from kwargs if it exists
        stat = kwargs.pop("stat", "density")

        # Plot each distribution
        for i, (name, data) in enumerate(data_dict.items()):
            if i < len(axes_flat):
                ax = axes_flat[i]
                color = colors[i % len(colors)]

                if kde_only:
                    sns.kdeplot(
                        data, fill=True, alpha=alpha, color=color, ax=ax, **kwargs
                    )
                else:
                    if hist:
                        sns.histplot(
                            data,
                            kde=kde,
                            alpha=alpha,
                            bins=bins,
                            stat=stat,  # Use the extracted stat parameter
                            color=color,
                            ax=ax,
                            **kwargs
                        )
                    elif kde:
                        sns.kdeplot(
                            data, fill=True, alpha=alpha, color=color, ax=ax, **kwargs
                        )

                if rug:
                    sns.rugplot(data, color=color, alpha=max(0.3, alpha - 0.3), ax=ax)

                # Set title for each subplot
                ax.set_title(name, fontsize=12, fontweight="bold")

                # Set appropriate y-label based on stat
                if hist:
                    if stat == "density":
                        y_label = "Density"
                    elif stat == "frequency":
                        y_label = "Frequency"
                    elif stat == "count":
                        y_label = "Count"
                    elif stat == "probability":
                        y_label = "Probability"
                    else:
                        y_label = stat.capitalize()

                    ax.set_ylabel(y_label, fontsize=12)

                ax.grid(alpha=0.2)

        # Hide unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        return fig, axes

    def plot_violin_box(
        self,
        data_dict,
        figsize=None,
        dpi=None,
        orient="v",
        jitter=True,
        palette=None,
        **kwargs
    ):
        """
        Create a violin plot with box plot overlay to compare distributions.

        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping distribution names to sample arrays
        figsize : tuple, optional
            Figure size (width, height) in inches
        dpi : int, optional
            Resolution in dots per inch for the figure. If None, uses the class default.
        orient : str
            Orientation ('v' for vertical, 'h' for horizontal)
        jitter : bool
            Whether to add jittered points for data visualization
        palette : str or list
            Color palette to use
        **kwargs :
            Additional keyword arguments for sns.violinplot or sns.boxplot

        Returns:
        --------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        if figsize is None:
            figsize = (12, 8) if len(data_dict) > 5 else self.figsize

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi or self.dpi)

        # Prepare data in long format
        data_long = []
        for name, values in data_dict.items():
            for v in values:
                data_long.append({"Distribution": name, "Value": v})
        df = pd.DataFrame(data_long)

        # Determine x and y based on orientation
        x, y = ("Distribution", "Value") if orient == "v" else ("Value", "Distribution")

        # Create violin plot
        sns.violinplot(
            x=x, y=y, data=df, inner="quartile", palette=palette, alpha=0.7, ax=ax
        )

        # Add jittered points if requested
        if jitter:
            sns.stripplot(
                x=x, y=y, data=df, size=2, color="black", alpha=0.3, jitter=True, ax=ax
            )

        # Customize plot appearance
        ax.set_title("Distribution Comparison", fontsize=14, fontweight="bold")
        ax.grid(axis="y" if orient == "v" else "x", alpha=0.3)

        plt.tight_layout()

        return ax

    def plot_pdf(
        self,
        pdf,
        x_eval,
        title="Probability Density Function",
        label="PDF",
        color=None,
        ax=None,
        dpi=None,
        **kwargs
    ):
        """
        Plots the Probability Density Function (PDF) for a given distribution.

        Parameters:
        -----------
        pdf : callable
            The probability density function to evaluate.
        x_eval : array-like
            The x values at which to evaluate the PDF.
        title : str
            The title of the plot.
        label : str
            The label for the PDF in the legend.
        color : str
            The color of the plot line.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes to plot on
        dpi : int, optional
            Resolution in dots per inch for the figure. If None, uses the class default.
        **kwargs :
            Additional keyword arguments for sns.lineplot

        Returns:
        --------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        # Create figure if ax not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=dpi or self.dpi)

        # Plot the PDF
        sns.lineplot(
            x=x_eval, y=pdf.evaluate(x_eval), label=label, color=color, ax=ax, **kwargs
        )

        # Customize plot appearance
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)

        plt.tight_layout()

        return ax

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=x_eval, y=pdf.evaluate(x_eval), label=label, color=color)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.show()
