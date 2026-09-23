"""
Visualization utilities for ANEMLL-Bench.

This module provides functions to visualize benchmark results using
matplotlib and other plotting libraries.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Dict, Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)


def plot_chip_comparison(
    chips: List[str],
    bandwidth: List[float],
    bandwidth_factor: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: str = 'Chip Performance Comparison',
    bandwidth_color: str = 'steelblue',
    **kwargs,
) -> plt.Figure:
    """
    Create a bar chart comparing memory bandwidth across different chips.

    Args:
        chips: List of chip names to display on x-axis
        bandwidth: List of memory bandwidth values in GB/s
        bandwidth_factor: Optional list of factors relative to baseline (e.g., '1.0x', '2.3x')
        figsize: Tuple specifying figure dimensions (width, height) in inches
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot (plt.show())
        title: Plot title
        bandwidth_color: Color for bandwidth bars

    Returns:
        Figure object containing the plot
    """
    if len(chips) != len(bandwidth):
        raise ValueError("Length of chips and bandwidth lists must be equal")

    if bandwidth_factor and len(bandwidth_factor) != len(chips):
        raise ValueError("Length of bandwidth_factor must match chips")

    # Setup positions and bar width
    x = np.arange(len(chips))
    width = 0.6

    # Create the figure
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot the bandwidth bars
    bars1 = ax1.bar(x, bandwidth, width, label='Bandwidth (GB/s)', color=bandwidth_color)
    ax1.margins(y=0.14)

    # Keep measured values inside the bars; counter-clockwise rotation keeps
    # all labels readable as chip count grows.
    for bar, value in zip(bars1, bandwidth):
        ax1.annotate(
            f'{value:.0f} GB/s',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, -5),
            textcoords='offset points',
            ha='center',
            va='top',
            rotation=90,
            fontsize=8,
            color='white',
        )

    # Configure the x-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(chips)
    ax1.set_xlabel('Chip')

    # Set y-axis label
    ax1.set_ylabel('Bandwidth (GB/s)\n(higher is better)', color=bandwidth_color)

    # Keep the benchmark caveat visible without competing with the title.
    fig.suptitle(title, y=0.985)
    fig.text(
        0.5,
        0.955,
        'Measured GB/s is slightly underestimated because benchmark timing includes compute work.',
        ha='center',
        va='top',
        fontsize=10,
        color='#555555',
    )
    ax1.legend(loc='upper left')

    # Annotate the bandwidth factor labels if provided
    if bandwidth_factor:
        for i, bar in enumerate(bars1):
            x_center = bar.get_x() + bar.get_width() / 2
            ax1.text(x_center, -0.08, bandwidth_factor[i],
                    ha='center', va='top',
                    transform=ax1.get_xaxis_transform(),
                    fontsize=8, color='black')

    # Check if any chip labels contain newlines and add more bottom padding if needed
    if any('\n' in chip for chip in chips):
        plt.subplots_adjust(bottom=0.18)  # Increase bottom margin for multi-line labels
    
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    
    # Save the figure if a path is provided
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {e}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
        
    return fig


def plot_benchmark_results(
    benchmark_data: Dict,
    model_name: Optional[str] = None,
    plot_type: str = 'comparison',
    plot_title: Optional[str] = None,
    save_dir: Optional[str] = None,
    show_plot: bool = True,
) -> Optional[str]:
    """
    Create visualizations from benchmark results.
    
    Args:
        benchmark_data: Dictionary containing benchmark results
        model_name: Optional name of the model for the title
        plot_type: Type of plot to generate ('comparison', 'timeline', etc.)
        plot_title: Optional custom plot title
        save_dir: Optional directory to save the generated figures
        show_plot: Whether to display the plots
        
    Returns:
        Path to the saved figure if save_dir is provided, otherwise None
    """
    # This is a placeholder function for future expansion
    # Currently just delegates to plot_chip_comparison
    
    if plot_type == 'comparison' and 'chips' in benchmark_data:
        # Extract data from benchmark_data
        chips = benchmark_data.get('chips', [])
        bandwidth = benchmark_data.get('bandwidth', [])
        bandwidth_factor = benchmark_data.get('bandwidth_factor', None)

        # Create title
        if plot_title:
            title = plot_title
        else:
            title = f'ANEMLL-BENCH: Apple Neural Engine Performance Comparison'

        if model_name:
            title += f' - {model_name}'

        # Create save path if directory is provided
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"chip_comparison{'_' + model_name if model_name else ''}.png"
            save_path = os.path.join(save_dir, filename)

        # Create the plot
        fig = plot_chip_comparison(
            chips=chips,
            bandwidth=bandwidth,
            bandwidth_factor=bandwidth_factor,
            title=title,
            save_path=save_path,
            show_plot=show_plot
        )
        
        return save_path
    
    else:
        logger.warning(f"Unsupported plot type: {plot_type}")
        return None
