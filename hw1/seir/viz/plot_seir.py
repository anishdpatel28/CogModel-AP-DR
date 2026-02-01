from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    infected: Union[np.ndarray, list],
    figsize: Tuple[float, float] = (10.0, 6.0),
    color: str = "#AA0000",
    linestyle: str = "dashed",
    marker: str = "o",
    xlabel: str = "Day",
    ylabel: str = "Number of Infected Cases",
    title: str = "Simulated Outbreak",
    xlabel_fontsize: int = 16,
    ylabel_fontsize: int = 16,
    title_fontsize: int = 20,
    grid_alpha: float = 0.2,
) -> plt.Figure:
    """Plots the time series of infected cases."""
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(infected, color=color, linestyle=linestyle, marker=marker)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(alpha=grid_alpha)
    return fig
