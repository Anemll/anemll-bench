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
    inference: List[float],
    bandwidth_factor: Optional[List[str]] = None,
    inference_factor: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: str = 'Chip Performance Comparison',
    bandwidth_color: str = 'steelblue',
    inference_color: str = 'indianred',
) -> plt.Figure:
    """
    Create a dual-axis bar chart comparing bandwidth and inference time across different chips.
    
    Args:
        chips: List of chip names to display on x-axis
        bandwidth: List of memory bandwidth values in GB/s
        inference: List of inference time values in ms
        bandwidth_factor: Optional list of factors relative to baseline (e.g., '1.0x', '2.3x')
        inference_factor: Optional list of factors relative to baseline (e.g., '1.0x', '2.0x')
        figsize: Tuple specifying figure dimensions (width, height) in inches
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot (plt.show())
        title: Plot title
        bandwidth_color: Color for bandwidth bars
        inference_color: Color for inference time bars
        
    Returns:
        Figure object containing the plot
    """
    if len(chips) != len(bandwidth) or len(chips) != len(inference):
        raise ValueError("Length of chips, bandwidth, and inference lists must be equal")
    
    if bandwidth_factor and len(bandwidth_factor) != len(chips):
        raise ValueError("Length of bandwidth_factor must match chips")
        
    if inference_factor and len(inference_factor) != len(chips):
        raise ValueError("Length of inference_factor must match chips")
    
    # Setup positions and bar width
    x = np.arange(len(chips))
    width = 0.35

    # Create the figure and twin axes
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Plot the bandwidth bars on ax1 (left y-axis)
    bars1 = ax1.bar(x - width/2, bandwidth, width, label='Bandwidth (GB/s)', color=bandwidth_color)

    # Plot the inference time bars on ax2 (right y-axis)
    bars2 = ax2.bar(x + width/2, inference, width, label='Inference Time (ms)', color=inference_color)

    # Configure the x-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(chips)
    ax1.set_xlabel('Chip')

    # Set y-axis labels
    ax1.set_ylabel('Bandwidth (GB/s)', color=bandwidth_color)
    ax2.set_ylabel('Inference Time (ms)', color=inference_color)

    # Set title and legends
    plt.title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Annotate the bandwidth factor labels if provided
    if bandwidth_factor:
        for i, bar in enumerate(bars1):
            x_center = bar.get_x() + bar.get_width() / 2
            ax1.text(x_center, -0.08, bandwidth_factor[i],
                    ha='center', va='top',
                    transform=ax1.get_xaxis_transform(),
                    fontsize=10, color='black')

    # Annotate the inference factor labels if provided
    if inference_factor:
        for i, bar in enumerate(bars2):
            x_center = bar.get_x() + bar.get_width() / 2
            ax2.text(x_center, -0.08, inference_factor[i],
                    ha='center', va='top',
                    transform=ax2.get_xaxis_transform(),
                    fontsize=10, color='black')

    plt.tight_layout()
    
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
        inference = benchmark_data.get('inference', [])
        bandwidth_factor = benchmark_data.get('bandwidth_factor', None)
        inference_factor = benchmark_data.get('inference_factor', None)
        
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
            inference=inference,
            bandwidth_factor=bandwidth_factor,
            inference_factor=inference_factor,
            title=title,
            save_path=save_path,
            show_plot=show_plot
        )
        
        return save_path
    
    else:
        logger.warning(f"Unsupported plot type: {plot_type}")
        return None 