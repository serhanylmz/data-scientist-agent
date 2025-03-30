import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_plot(df, plot_type, output_dir='./output/plots', **kwargs):
    """
    Generate a plot for the given DataFrame.
    
    Args:
        df (DataFrame): The pandas DataFrame to plot
        plot_type (str): Type of plot to generate ('line', 'bar', 'scatter', 'histogram', 'boxplot', 'heatmap')
        output_dir (str, optional): Directory to save the plot
        **kwargs: Additional arguments for the specific plot type
            
    Returns:
        tuple: (plot file path, str message)
    """
    if df is None or df.empty:
        return None, "Error: DataFrame is empty or None"

    # Ensure plot_type is a string without quotes
    if isinstance(plot_type, str):
        plot_type = plot_type.strip("'\"")
    
    # Convert string representations of dictionaries in kwargs to actual dictionaries
    for key, value in list(kwargs.items()):
        if isinstance(value, str):
            # Try to evaluate as a Python literal (dict, list, etc.)
            if (value.startswith('{') and value.endswith('}')) or \
               (value.startswith('[') and value.endswith(']')):
                try:
                    import ast
                    kwargs[key] = ast.literal_eval(value)
                except:
                    # Keep as string if evaluation fails
                    pass
            # Remove quotes from simple string values
            elif value.startswith(("'", '"')) and value.endswith(("'", '"')):
                kwargs[key] = value.strip("'\"")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename for the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plot_type}_{timestamp}.png"
    file_path = os.path.join(output_dir, filename)
    
    # Create a figure with a specified size
    fig_size = kwargs.get('figsize', (10, 6))
    fig, ax = plt.subplots(figsize=fig_size)
    
    try:
        # Set title and labels
        title = kwargs.get('title', f'{plot_type.capitalize()} Plot')
        xlabel = kwargs.get('xlabel', '')
        ylabel = kwargs.get('ylabel', '')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # Generate the specified plot type
        if plot_type == 'line':
            _line_plot(df, ax, **kwargs)
        elif plot_type == 'bar':
            _bar_plot(df, ax, **kwargs)
        elif plot_type == 'scatter':
            _scatter_plot(df, ax, **kwargs)
        elif plot_type == 'histogram':
            _histogram_plot(df, ax, **kwargs)
        elif plot_type == 'boxplot':
            _boxplot(df, ax, **kwargs)
        elif plot_type == 'heatmap':
            _heatmap(df, ax, **kwargs)
        else:
            return None, f"Error: Unsupported plot type '{plot_type}'"
        
        # Save the plot to a file
        plt.tight_layout()
        plt.savefig(file_path, dpi=kwargs.get('dpi', 300))
        plt.close(fig)
        
        message = f"Generated {plot_type} plot: {file_path}"
        return file_path, message
    
    except Exception as e:
        plt.close(fig)
        return None, f"Error generating {plot_type} plot: {str(e)}"

def _line_plot(df, ax, **kwargs):
    """Generate a line plot."""
    x = kwargs.get('x')
    y = kwargs.get('y')
    
    if x and y:
        if isinstance(y, list):
            for col in y:
                df.plot(x=x, y=col, ax=ax, label=col)
        else:
            df.plot(x=x, y=y, ax=ax, label=y)
    else:
        df.plot(ax=ax)
    
    if kwargs.get('grid', True):
        ax.grid(True, alpha=0.3)

def _bar_plot(df, ax, **kwargs):
    """Generate a bar plot."""
    x = kwargs.get('x')
    y = kwargs.get('y')
    
    if x and y:
        df.plot.bar(x=x, y=y, ax=ax)
    else:
        df.plot.bar(ax=ax)
    
    if kwargs.get('grid', True):
        ax.grid(True, alpha=0.3, axis='y')

def _scatter_plot(df, ax, **kwargs):
    """Generate a scatter plot."""
    x = kwargs.get('x')
    y = kwargs.get('y')
    
    if not (x and y):
        return None, "Error: Both 'x' and 'y' columns must be specified for a scatter plot"
    
    c = kwargs.get('color_by')
    if c and c in df.columns:
        scatter = ax.scatter(df[x], df[y], c=df[c], cmap=kwargs.get('cmap', 'viridis'), alpha=0.6)
        plt.colorbar(scatter, ax=ax, label=c)
    else:
        ax.scatter(df[x], df[y], alpha=0.6)
    
    if kwargs.get('grid', True):
        ax.grid(True, alpha=0.3)

def _histogram_plot(df, ax, **kwargs):
    """Generate a histogram."""
    column = kwargs.get('column')
    
    if not column:
        return None, "Error: 'column' must be specified for a histogram"
    
    bins = kwargs.get('bins', 10)
    df[column].hist(ax=ax, bins=bins, alpha=0.7)
    
    if kwargs.get('kde', False):
        df[column].plot.kde(ax=ax)
    
    if kwargs.get('grid', True):
        ax.grid(True, alpha=0.3)

def _boxplot(df, ax, **kwargs):
    """Generate a boxplot."""
    columns = kwargs.get('columns')
    
    if columns:
        if isinstance(columns, list):
            df[columns].boxplot(ax=ax)
        else:
            df[[columns]].boxplot(ax=ax)
    else:
        df.boxplot(ax=ax)
    
    if kwargs.get('grid', True):
        ax.grid(True, alpha=0.3, axis='y')

def _heatmap(df, ax, **kwargs):
    """Generate a heatmap (e.g., for correlation matrices)."""
    cmap = kwargs.get('cmap', 'coolwarm')
    
    im = ax.imshow(df, cmap=cmap)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add tick marks and labels
    tick_spacing = kwargs.get('tick_spacing', 1)
    ax.set_xticks(np.arange(0, len(df.columns), tick_spacing))
    ax.set_yticks(np.arange(0, len(df.index), tick_spacing))
    ax.set_xticklabels(df.columns[::tick_spacing], rotation=90)
    ax.set_yticklabels(df.index[::tick_spacing])
    
    # Add values in each cell if requested
    if kwargs.get('annot', False):
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                ax.text(j, i, f"{df.iloc[i, j]:.2f}", ha="center", va="center", color="black")

def plot_multiple(df, plot_configs, output_dir='./output/plots'):
    """
    Generate multiple plots from a list of configurations.
    
    Args:
        df (DataFrame): The pandas DataFrame to plot
        plot_configs (list): List of dictionaries, each containing plot configuration
        output_dir (str, optional): Directory to save the plots
            
    Returns:
        tuple: (list of plot file paths, str message)
    """
    if df is None or df.empty:
        return None, "Error: DataFrame is empty or None"
    
    if not plot_configs:
        return None, "Error: No plot configurations provided"
    
    file_paths = []
    error_messages = []
    
    for config in plot_configs:
        plot_type = config.pop('plot_type', None)
        if not plot_type:
            error_messages.append("Missing plot_type in configuration")
            continue
        
        file_path, message = generate_plot(df, plot_type, output_dir, **config)
        if file_path:
            file_paths.append(file_path)
        else:
            error_messages.append(message)
    
    message = f"Generated {len(file_paths)} plots"
    if error_messages:
        message += f" with {len(error_messages)} errors: {'; '.join(error_messages)}"
    
    return file_paths, message 