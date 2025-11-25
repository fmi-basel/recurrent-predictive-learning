import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import pandas as pd

import wandb

import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_rgb


# Set the global default fontsize for tick labels
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

method_names = {
    'sup': 'Sup.',
    'rand': 'Rand.',
    'raw': "Pixel",
    'pred': 'RPL',
    'inv_sg': 'IL',
    'inv_sg_nr': r'$\text{IL}^{\text{nr}}$',
    'pred-mem': 'CL',
    'invZ': r'$\text{IL}^z$',
    'invC': r'$\text{IL}^c$',
}
method_colors = {
    'sup': 'black',
    'rand': 'gray',
    'raw': '#7B1FA2',
    'pred': '#994455',
    'inv_sg': '#888888',
    'inv_sg_nr': '#5f3f2a',
    'pred-mem': '#004488',
    'invZ': '#e28f5a',
    'invC': '#5aade2',
}
baseline_styles = {'sup': ('black', '-'), 'rand': ('gray', '--'), 'raw': ('#7B1FA2', ':')}
true_var_names = [r'$r_1$', r'$r_2$', r'$r_3$',
                  r'$v_1$', r'$v_2$', r'$v_3$', r'$a$', r'$b$']

# Define linestyles for baselines
baseline_styles = {'sup': ('black', '-'), 'rand': ('gray', '--'), 'raw': ('#7B1FA2', ':')}

def plot_advanced_comparison(results_df, methods_to_plot, baseline_methods, metric_configs,
                          method_names, method_colors=None, figsize=None, dataset='mnist_hmm', 
                          ylim=None, loc_legend='best', display_legend=True, display_xticks=True, 
                          display_yticks=True, legend_ax_idx=-1, shared_y=True, plot_spacing=0.3,
                          legend_ncols=1, sharey=False, point_size=25, task_width_factor=1.0):
    """Create advanced comparison plots for different methods across tasks."""
    # Filter data for selected dataset
    data = results_df[results_df['dataset'] == dataset].copy()

    # If sharey is False, use the original plotting behavior
    if not sharey:
        return _plot_advanced_comparison_original(
            results_df, methods_to_plot, baseline_methods, metric_configs, method_names,
            method_colors, figsize, dataset, ylim, loc_legend, display_legend, display_xticks,
            display_yticks, legend_ax_idx, shared_y, plot_spacing, False, point_size, task_width_factor
        )

    # Calculate how many different metric types we have
    metric_types = list(metric_configs.keys())
    num_metrics = len(metric_types)

    # Find maximum number of tasks for consistent scaling
    max_num_tasks = max(
        (len(config.get('individual_tasks', {})) + len(config.get('averaged_tasks', [])))
        for metric_type, config in metric_configs.items()
    )

    # Calculate appropriate figure size if not provided
    if figsize is None:
        total_tasks = sum(len(metric_configs[m].get('individual_tasks', {})) + 
                           len(metric_configs[m].get('averaged_tasks', [])) 
                           for m in metric_types)
        width = min(16, max(7, 1.5 * total_tasks))
        height = 4
        figsize = (width, height)

    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=figsize)

    # Calculate width ratios based on number of tasks only
    # This makes each section's width proportional to its task count
    width_ratios = []
    for metric_type in metric_types:
        config = metric_configs[metric_type]
        num_tasks = len(config.get('individual_tasks', {})) + len(config.get('averaged_tasks', []))
        # Width ratio simply proportional to task count
        ratio = max(num_tasks, 1)
        width_ratios.append(ratio)

    # Create GridSpec without extra column for legend
    gs = gridspec.GridSpec(1, num_metrics, width_ratios=width_ratios, wspace=plot_spacing)

    # Create axes array and track legend elements
    axes = []
    all_legend_elements = []

    # Process each metric type
    for metric_idx, metric_type in enumerate(metric_types):
        config = metric_configs[metric_type]

        # Create axis for this metric
        ax = fig.add_subplot(gs[0, metric_idx])
        axes.append(ax)

        # Collect task data
        task_configs = []
        plot_data = data[data['method'].isin(methods_to_plot)].copy()
        baseline_data = data[data['method'].isin(baseline_methods)].copy()

        # Process individual tasks
        for task, display_name in config.get('individual_tasks', {}).items():
            task_data = plot_data[plot_data['task'] == task].copy()
            task_baselines = baseline_data[baseline_data['task'] == task].copy()

            if len(task_data) > 0:
                task_configs.append({
                    'task_id': task,
                    'display_name': display_name,
                    'data': task_data,
                    'baselines': task_baselines,
                    'type': 'individual'
                })

        # Process averaged tasks
        for i, avg_config in enumerate(config.get('averaged_tasks', [])):
            tasks_to_avg = avg_config['tasks']
            avg_name = avg_config['name']

            valid_tasks = [t for t in tasks_to_avg if t in plot_data['task'].unique()]

            if valid_tasks:
                # Filter and compute averages
                avg_data = plot_data[plot_data['task'].isin(valid_tasks)].copy()
                avg_baseline = baseline_data[baseline_data['task'].isin(valid_tasks)].copy()

                avg_plot_data = avg_data.groupby(['method', 'seed'])[metric_type].mean().reset_index()
                avg_plot_data['task'] = f'avg_{i}'

                avg_baseline_data = avg_baseline.groupby(['method', 'seed'])[metric_type].mean().reset_index()
                avg_baseline_data['task'] = f'avg_{i}'

                task_configs.append({
                    'task_id': f'avg_{i}',
                    'display_name': avg_name,
                    'data': avg_plot_data,
                    'baselines': avg_baseline_data,
                    'type': 'averaged'
                })

        # Calculate global y-limits
        y_global_min, y_global_max = float('inf'), float('-inf')

        # Use user-defined limits if available
        if ylim and metric_type in ylim:
            y_global_min, y_global_max = ylim[metric_type]
        else:
            # Calculate from data
            for task_config in task_configs:
                task_data = task_config['data']
                task_baselines = task_config['baselines']

                if not task_data.empty:
                    y_global_min = min(y_global_min, task_data[metric_type].min())
                    y_global_max = max(y_global_max, task_data[metric_type].max())

                if not task_baselines.empty:
                    y_global_min = min(y_global_min, task_baselines[metric_type].min())
                    y_global_max = max(y_global_max, task_baselines[metric_type].max())

            # Add padding
            if y_global_min != float('inf'):
                padding = (y_global_max - y_global_min) * 0.1
                y_global_min -= padding
                y_global_max += padding

        # Check if this is the last metric
        is_last_metric = (metric_idx == len(metric_types) - 1)

        # Keep task_width_factor consistent across all metrics
        # This ensures each task section has the same width
        num_tasks = len(task_configs)
        current_task_width_factor = task_width_factor

        # Create the combined plot
        if task_configs:
            ax, legend_elements = _plot_combined_tasks(
                ax=ax,
                task_configs=task_configs,
                methods_to_plot=methods_to_plot,
                method_names=method_names,
                method_colors=method_colors,
                metric_type=metric_type,
                baseline_styles=baseline_styles,
                baseline_methods=baseline_methods,
                y_limits=(y_global_min, y_global_max) if y_global_min != float('inf') else None,
                display_legend=display_legend,
                loc_legend=loc_legend,
                is_last_metric=is_last_metric,
                point_size=point_size,
                task_width_factor=current_task_width_factor
            )

            # Save legend elements from last metric
            if is_last_metric:
                all_legend_elements = legend_elements

            # Set y-label
            if display_yticks:
                if metric_type == 'accuracy':
                    ax.set_ylabel('Acc.', fontsize=8)
                elif metric_type == 'r2':
                    ax.set_ylabel(r'$R^2$', fontsize=8)
                else:
                    ax.set_ylabel(metric_type.upper(), fontsize=8)
            else:
                ax.set_yticks([])
                ax.set_ylabel('')

    # Add legend to a specific axis based on legend_ax_idx
    if display_legend and all_legend_elements:
        # Default to last axis if legend_ax_idx is -1 or out of range
        if legend_ax_idx == -1 or legend_ax_idx >= len(axes):
            legend_ax = axes[-1]
        else:
            legend_ax = axes[legend_ax_idx]

        # Create method labels list
        method_labels = [method_names.get(method, method) for method in methods_to_plot]

        # Extract baseline labels
        baseline_labels = [elem.get_label() for elem in all_legend_elements if isinstance(elem, Line2D)]

        # Create legend
        legend = legend_ax.legend(
            all_legend_elements,
            method_labels + baseline_labels,
            loc=loc_legend,
            fontsize=6,
            frameon=False,
            handlelength=1.5,
            handletextpad=0.4,
            labelspacing=0.4,
            ncols=legend_ncols
        )

    return fig, axes


def _plot_combined_tasks(ax, task_configs, methods_to_plot, method_names, method_colors, 
                        metric_type, baseline_styles, baseline_methods, y_limits=None, 
                        display_legend=True, loc_legend='best', is_last_metric=False,
                        point_size=25, task_width_factor=1.0):
    """Helper function to create a combined plot with all tasks on the x-axis."""
    if not task_configs:
        return ax, []

    # Calculate positions for tasks and methods
    num_tasks = len(task_configs)
    num_methods = len(methods_to_plot)

    # Calculate task width and spacing
    task_width = (num_methods + 1) * task_width_factor
    method_positions = {}

    # Set up x-ticks and positions
    task_centers = []
    task_labels = []
    all_x_positions = []

    for task_idx, task_config in enumerate(task_configs):
        display_name = task_config['display_name']

        # Calculate center position for this task
        task_center = task_idx * task_width + (task_width / 2) - 0.6
        task_centers.append(task_center)
        task_labels.append(display_name)

        # Calculate method positions within this task
        for method_idx, method in enumerate(methods_to_plot):
            x_pos = task_idx * task_width + method_idx * task_width_factor + 0.5 * task_width_factor
            all_x_positions.append(x_pos)
            method_positions[(task_config['task_id'], method)] = x_pos

    # Set axis limits based on actual content
    if all_x_positions:
        # Add padding proportional to task_width_factor
        left_padding = task_width_factor * 0.75
        right_padding = task_width_factor * 0.75

        leftmost = min(all_x_positions) - left_padding
        rightmost = max(all_x_positions) + right_padding

        ax.set_xlim(leftmost, rightmost)
    else:
        # Fallback to original calculation
        padding = task_width * 0.1
        ax.set_xlim(-padding, num_tasks * task_width - padding)

    if y_limits:
        ax.set_ylim(y_limits)

    ax.set_xticks([])  # Remove x-ticks
    add_string_formatters_to_axes(ax)

    # Add horizontal labels at the center of each section
    for center, label in zip(task_centers, task_labels):
        ax.text(center, -0.1, label, ha='center', va='top', transform=ax.get_xaxis_transform(),
                fontsize=8, rotation=0)

    # Draw grid lines to separate tasks
    for i in range(1, num_tasks):
        x_pos = i * task_width - 0.5
        ax.axvline(x=x_pos, color='lightgray', linestyle='-', alpha=0.5, zorder=0)

    # Plot data points for each task and method
    for task_config in task_configs:
        task_id = task_config['task_id']
        task_data = task_config['data']

        # Plot method points for this task
        for method in methods_to_plot:
            method_data = task_data[task_data['method'] == method]

            if len(method_data) > 0:
                # Get position and values
                x_pos = method_positions.get((task_id, method))
                if x_pos is None:
                    continue

                y_values = method_data[metric_type].values
                color = method_colors.get(method, 'black') if method_colors else 'black'

                # Add small offsets to avoid overlaps
                point_offsets = np.linspace(-0.2, 0.2, len(y_values))
                np.random.shuffle(point_offsets)

                # Plot points
                ax.scatter(
                    [x_pos + offset for offset in point_offsets],
                    y_values,
                    color=color,
                    s=point_size,
                    alpha=0.8,
                    zorder=10
                )

                # Calculate and plot mean
                if len(y_values) > 0:
                    mean_value = np.mean(y_values)
                    ax.plot(
                        [x_pos - 0.4, x_pos + 0.4],
                        [mean_value, mean_value],
                        color=color,
                        linewidth=1,
                        zorder=15
                    )

    # Plot baseline lines for each task section
    for task_idx, task_config in enumerate(task_configs):
        task_id = task_config['task_id']
        task_baselines = task_config['baselines']

        # Get all method positions for this task
        task_method_positions = [pos for (tid, met), pos in method_positions.items() if tid == task_id]

        if task_method_positions:
            # Calculate x-range from method positions
            x_start = min(task_method_positions) - 0.6  # A bit before the first method
            x_end = max(task_method_positions) + 0.6    # A bit after the last method

            # Add baseline lines
            for baseline_method in baseline_methods:
                baseline_data = task_baselines[task_baselines['method'] == baseline_method]

                if len(baseline_data) > 0:
                    baseline_value = baseline_data[metric_type].mean()

                    if baseline_method in baseline_styles:
                        color, style = baseline_styles[baseline_method]
                        label = method_names.get(baseline_method, baseline_method)

                        # Only add label for the first task
                        ax.plot(
                            [x_start, x_end],
                            [baseline_value, baseline_value],
                            color=color,
                            linestyle=style,
                            label=label if task_idx == 0 else None,
                            linewidth=0.5,
                            zorder=5
                        )

    # Prepare legend elements if this is the last metric plot
    legend_elements = []
    if is_last_metric:
        # Add method points to legend
        for method in methods_to_plot:
            if method_colors and method in method_colors:
                color = method_colors[method]
                label = method_names.get(method, method)
                legend_elements.append(Patch(facecolor=color, edgecolor='none', label=label))

        # Get baseline elements
        handles, labels = ax.get_legend_handles_labels()
        legend_elements.extend(handles)

    sns.despine(ax=ax)

    return ax, legend_elements


def _plot_single_task(ax, task_data, task_baselines, methods_to_plot, method_names, 
                      display_name, metric_type, baseline_styles, y_limits=None, 
                      show_ylabel=True, show_yticks=True, method_colors=None, point_size=25):
    """Plot single task data with baseline comparisons."""
    # Set title and labels
    ax.set_title(display_name, fontsize=8)
    
    if show_ylabel:
        if metric_type == 'accuracy':
            ax.set_ylabel('Acc.', fontsize=8)
        else:
            ax.set_ylabel(metric_type.upper(), fontsize=8)
    
    # Set y limits if provided
    if y_limits:
        ax.set_ylim(y_limits)
    
    # Hide x-ticks and labels
    ax.set_xticks([])
    
    # Manage y-ticks
    if not show_yticks:
        ax.set_yticks([])
    
    # Plot methods
    x_positions = {}
    num_methods = len(methods_to_plot)
    x_start = 0.5
    x_end = num_methods + 0.5
    
    # Plot method data points
    for i, method in enumerate(methods_to_plot):
        x_pos = x_start + i
        x_positions[method] = x_pos
        
        method_data = task_data[task_data['method'] == method]
        
        if not method_data.empty:
            y_values = method_data[metric_type].values
            
            # Use method color if available
            color = method_colors.get(method, 'C'+str(i % 10)) if method_colors else 'C'+str(i % 10)
            
            # Add small offsets to avoid overlaps
            point_offsets = np.linspace(-0.2, 0.2, len(y_values))
            np.random.shuffle(point_offsets)
            
            # Plot points
            ax.scatter(
                [x_pos + offset for offset in point_offsets],
                y_values,
                color=color,
                s=point_size,
                alpha=0.8,
                zorder=10
            )
            
            # Calculate and plot mean
            if len(y_values) > 0:
                mean_value = np.mean(y_values)
                ax.plot(
                    [x_pos - 0.3, x_pos + 0.3],
                    [mean_value, mean_value],
                    color=color,
                    linewidth=2,
                    zorder=15
                )
    
    # Plot baseline methods
    for baseline_method, (color, style) in baseline_styles.items():
        baseline_data = task_baselines[task_baselines['method'] == baseline_method]
        
        if not baseline_data.empty:
            baseline_value = baseline_data[metric_type].mean()
            label = method_names.get(baseline_method, baseline_method)
            
            ax.axhline(
                y=baseline_value,
                color=color,
                linestyle=style,
                label=label,
                linewidth=0.5,
                alpha=0.7,
                zorder=5
            )
    
    # Set x-axis limits
    ax.set_xlim(x_start - 0.5, x_end - 0.5)

    sns.despine(ax=ax)
    
    return ax


def _plot_advanced_comparison_original(results_df, methods_to_plot, baseline_methods, metric_configs,
                                      method_names, method_colors=None, figsize=None, dataset='mnist_hmm', 
                                      ylim=None, loc_legend='best', display_legend=True, display_xticks=True, 
                                      display_yticks=True, legend_ax_idx=-1, shared_y=True, plot_spacing=0.3,
                                      sharey=False, point_size=25, task_width_factor=1.0):
    """Plot advanced comparison plots for different methods."""
    # Filter data for selected dataset
    data = results_df[results_df['dataset'] == dataset].copy()
    
    # Calculate the total number of plots needed
    total_plots = 0
    metric_to_plots = {}
    for metric_type, config in metric_configs.items():
        individual_count = len(config.get('individual_tasks', {}))
        averaged_count = len(config.get('averaged_tasks', []))
        metric_to_plots[metric_type] = individual_count + averaged_count
        total_plots += individual_count + averaged_count
    
    # Calculate appropriate figure size if not provided
    if figsize is None:
        width = min(16, max(3.5, 1.5 * total_plots))
        height = 3
        figsize = (width, height)
    
    # Single row layout
    rows = 1
    total_cols = total_plots
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    
    # Track metric positions
    metric_types = [m for m, count in metric_to_plots.items() if count > 0]
    metric_positions = []
    current_pos = 0
    
    for metric_type in sorted(metric_types):
        metric_positions.append((metric_type, current_pos, current_pos + metric_to_plots[metric_type] - 1))
        current_pos += metric_to_plots[metric_type]
    
    # Add extra space between metric types
    if len(metric_types) > 1:
        width_ratios = []
        for i in range(total_cols):
            width_ratios.append(1)
            
            # Add spacer between metric groups
            for metric_type, start_pos, end_pos in metric_positions[:-1]:
                if i == end_pos:
                    width_ratios.append(0.3)
                    total_cols += 1
        
        gs = gridspec.GridSpec(rows, total_cols, wspace=plot_spacing, width_ratios=width_ratios)
    else:
        gs = gridspec.GridSpec(rows, total_cols, wspace=plot_spacing)
    
    # Create axes array
    all_axes = [[None for _ in range(total_cols)] for _ in range(rows)]
    
    # Track metric data
    metric_axes = {}
    metric_data = {}
    metric_baselines = {}
    
    # Process each metric type
    col_offset = 0
    spacer_count = 0
    
    for metric_type, config in sorted(metric_configs.items()):
        if metric_to_plots[metric_type] == 0:
            continue
        
        # Initialize data structures
        metric_axes[metric_type] = []
        metric_data[metric_type] = []
        metric_baselines[metric_type] = []
        
        all_tasks = []
        processed_tasks = 0
        
        # Filter for selected methods
        plot_data = data[data['method'].isin(methods_to_plot)].copy()
        baseline_data = data[data['method'].isin(baseline_methods)].copy()
        
        # Process individual tasks
        for task, display_name in config.get('individual_tasks', {}).items():
            task_data = plot_data[plot_data['task'] == task].copy()
            
            if len(task_data) == 0:
                continue
            
            # Process baseline data
            task_baselines = baseline_data[baseline_data['task'] == task].copy()
            
            # Store data
            metric_data[metric_type].append(task_data)
            metric_baselines[metric_type].append(task_baselines)
            
            # Create axis
            col_idx = col_offset + processed_tasks + spacer_count
            ax = fig.add_subplot(gs[0, col_idx])
            all_axes[0][col_idx] = ax
            metric_axes[metric_type].append((ax, display_name))
            
            processed_tasks += 1
            all_tasks.append(task)
        
        # Process averaged tasks
        for i, avg_config in enumerate(config.get('averaged_tasks', [])):
            tasks_to_avg = avg_config['tasks']
            avg_name = avg_config['name']
            
            valid_tasks = [t for t in tasks_to_avg if t in plot_data['task'].unique()]
            
            if not valid_tasks:
                continue
            
            # Filter and compute averages
            avg_data = plot_data[plot_data['task'].isin(valid_tasks)].copy()
            avg_baseline = baseline_data[baseline_data['task'].isin(valid_tasks)].copy()
            
            avg_plot_data = avg_data.groupby(['method', 'seed'])[metric_type].mean().reset_index()
            avg_plot_data['task'] = f'avg_{i}'
            
            avg_baseline_data = avg_baseline.groupby(['method', 'seed'])[metric_type].mean().reset_index()
            avg_baseline_data['task'] = f'avg_{i}'
            
            # Store data
            metric_data[metric_type].append(avg_plot_data)
            metric_baselines[metric_type].append(avg_baseline_data)
            
            # Create axis
            col_idx = col_offset + processed_tasks + spacer_count
            ax = fig.add_subplot(gs[0, col_idx])
            all_axes[0][col_idx] = ax
            metric_axes[metric_type].append((ax, avg_name))
            
            processed_tasks += 1
        
        # Update position trackers
        col_offset += processed_tasks
        if metric_type != sorted(metric_configs.keys())[-1] and processed_tasks > 0:
            spacer_count += 1
    
    # Plot data with appropriate limits
    for metric_type, axes_list in metric_axes.items():
        # Calculate global y range if needed
        y_global_min, y_global_max = float('inf'), float('-inf')
        
        # Use user-defined limits if available
        if ylim and metric_type in ylim:
            y_global_min, y_global_max = ylim[metric_type]
        # Otherwise calculate from data if shared_y
        elif shared_y:
            # From data
            for task_data in metric_data[metric_type]:
                if not task_data.empty:
                    y_global_min = min(y_global_min, task_data[metric_type].min())
                    y_global_max = max(y_global_max, task_data[metric_type].max())
            
            # From baselines
            for task_baselines in metric_baselines[metric_type]:
                if not task_baselines.empty:
                    y_global_min = min(y_global_min, task_baselines[metric_type].min())
                    y_global_max = max(y_global_max, task_baselines[metric_type].max())
            
            # Add padding
            if y_global_min != float('inf'):
                padding = (y_global_max - y_global_min) * 0.1
                y_global_min -= padding
                y_global_max += padding
        
        # Check for individual y-limits
        has_individual_ylims = 'ylim_individual' in metric_configs[metric_type]
        has_averaged_ylims = 'averaged_tasks' in metric_configs[metric_type] and 'ylim_averaged' in metric_configs[metric_type]
        
        # Track indices
        individual_idx = 0
        averaged_idx = 0
        
        # Now plot each task with appropriate limits
        for idx, ((ax, display_name), task_data, task_baselines) in enumerate(zip(
            axes_list, metric_data[metric_type], metric_baselines[metric_type])):
            
            # Determine if first axis (for y-label)
            is_first = idx == 0
            
            # Get task type and set y-limits
            task_key = task_data['task'].iloc[0] if not task_data.empty else None
            
            # Default to shared limits
            y_limits = None
            if shared_y and y_global_min != float('inf'):
                y_limits = (y_global_min, y_global_max)
            
            # Check for per-task limits
            if task_key:
                if task_key.startswith('avg_'):
                    # Averaged task
                    if has_averaged_ylims and averaged_idx in metric_configs[metric_type]['ylim_averaged']:
                        y_limits = metric_configs[metric_type]['ylim_averaged'][averaged_idx]
                    averaged_idx += 1
                else:
                    # Individual task
                    if has_individual_ylims and task_key in metric_configs[metric_type]['ylim_individual']:
                        y_limits = metric_configs[metric_type]['ylim_individual'][task_key]
                    individual_idx += 1
            
            # Calculate limits from data if not set
            if y_limits is None and not shared_y:
                y_min, y_max = float('inf'), float('-inf')
                
                if not task_data.empty:
                    y_min = min(y_min, task_data[metric_type].min())
                    y_max = max(y_max, task_data[metric_type].max())
                
                if not task_baselines.empty:
                    y_min = min(y_min, task_baselines[metric_type].min())
                    y_max = max(y_max, task_baselines[metric_type].max())
                
                if y_min != float('inf'):
                    padding = (y_max - y_min) * 0.1
                    y_limits = (y_min - padding, y_max + padding)
            
            # Determine label visibility
            show_ylabel = is_first and display_yticks
            show_yticks = display_yticks
            
            # Plot the task
            _plot_single_task(
                ax, task_data, task_baselines, methods_to_plot, method_names,
                display_name, metric_type, baseline_styles, y_limits=y_limits,
                show_ylabel=show_ylabel, show_yticks=show_yticks,
                method_colors=method_colors, point_size=point_size
            )
    
    # Add legend if requested
    if display_legend:
        # Find the appropriate axis for legend based on legend_ax_idx
        if legend_ax_idx == -1 or legend_ax_idx >= len(all_axes[0]):
            # Use the last visible axis if invalid index
            legend_ax = None
            for ax in reversed(all_axes[0]):
                if ax is not None and ax.get_visible():
                    legend_ax = ax
                    break
        else:
            # Use the specified axis
            legend_ax = all_axes[0][legend_ax_idx]
        
        if legend_ax:
            handles, labels = legend_ax.get_legend_handles_labels()
            
            # Add method colors to legend
            if method_colors:
                method_handles = []
                method_labels = []
                
                for method in methods_to_plot:
                    if method in method_colors:
                        color = method_colors[method]
                        label = method_names.get(method, method)
                        method_handles.append(Patch(facecolor=color, edgecolor='none', label=label))
                        method_labels.append(label)
                
                # Combine with baseline handles and labels
                handles = method_handles + handles
                labels = method_labels + labels
            
            # Create legend with specified location
            legend_ax.legend(handles, labels,
                     loc=loc_legend,
                     fontsize=6,
                     frameon=False,
                     handlelength=1,
                     handletextpad=0.4,
                     labelspacing=0.2)
    
    sns.despine()
    return fig, all_axes

def plot_comparison(results_df, methods_to_plot, baseline_methods, task_names, method_names, 
                   method_colors=None, task_names_to_average=None, averaged_task_name=None,
                   figsize=(3.5, 3), dataset='mnist_hmm', ylim=None, loc_legend='best',
                   display_legend=True, display_xticks=True, display_yticks=True, legend_ax_idx=-1,
                   legend_ncols=1, sharey=False, point_size=25, task_width_factor=1.0):
    """Plot comparison plots for different methods."""
    # Convert old format to new format
    metric_configs = {
        'accuracy': {
            'individual_tasks': {}
        }
    }
    
    # Add individual tasks
    metric_configs['accuracy']['individual_tasks'] = task_names.copy()
    
    # Add averaged tasks if specified
    if task_names_to_average and averaged_task_name:
        metric_configs['accuracy']['averaged_tasks'] = [{
            'tasks': task_names_to_average,
            'name': averaged_task_name
        }]
    
    # Convert ylim to new format
    ylim_dict = None
    if ylim:
        # Check if ylim is a tuple or dict
        if isinstance(ylim, tuple):
            ylim_dict = {'accuracy': ylim}
        else:
            ylim_dict = ylim
    
    # Call the new function
    return plot_advanced_comparison(
        results_df=results_df,
        methods_to_plot=methods_to_plot,
        baseline_methods=baseline_methods,
        metric_configs=metric_configs,
        method_names=method_names,
        method_colors=method_colors,
        figsize=figsize,
        dataset=dataset,
        ylim=ylim_dict,
        loc_legend=loc_legend,
        display_legend=display_legend,
        legend_ncols=legend_ncols,
        display_xticks=display_xticks,
        display_yticks=display_yticks,
        legend_ax_idx=legend_ax_idx,
        sharey=sharey,
        point_size=point_size,
        task_width_factor=task_width_factor
    )


# Configure matplotlib to use LaTeX for r'' strings while keeping other text unchanged
mpl.rcParams.update({
    'text.usetex': True,          # Use LaTeX for r'' strings
    'font.family': 'sans-serif',  # Default font family for regular text
    'mathtext.fontset': 'dejavusans',  # Font for math not using r''
    'axes.formatter.use_mathtext': False  # Enable math text for tick labels
})


def extract_all_results(datasets, method_map, api=None, wandb_project="RPL", remove_offline=False):
    """Extract all results from WandB project.

    Args:
        datasets: List of dataset names (e.g. ['mnist_hmm', 'mnist_wm'])
        method_map: Dict mapping method names to their keys 
        api: Optional wandb.Api instance
        wandb_project: Name of the WandB project to query

    Returns:
        Nested dict: {dataset: {metric_type: {method: {task: [values]}}}}
    """
    if api is None:
        api = wandb.Api()

    results = {
        dataset: {
            method: {} for method in method_map.keys()
        } for dataset in datasets
    }

    # Debug counter
    processed_runs = 0
    acc_metrics_found = 0
    r2_metrics_found = 0

    runs = api.runs(wandb_project, per_page=1000)

    for run in runs:
        try:
            *name_parts, seed_str = run.name.split('_')
            try:
                seed = int(seed_str)
            except ValueError:
                continue

            if "offline" in name_parts and remove_offline:
                name_parts.remove("offline")

            run_name = '_'.join(name_parts)

            # Sort datasets by length in descending order to match the longest prefix first
            sorted_datasets = sorted(datasets, key=len, reverse=True)
            matching_dataset = next((d for d in sorted_datasets if run_name.startswith(d)), None)

            if matching_dataset is None:
                continue

            method_key = run_name[len(matching_dataset)+1:]
            # Use exact matching
            method = next((m for m, k in method_map.items() if method_key == k), None)
            if method is None:
                print(f"No exact match found for method_key: {method_key}")
                continue

            last_metrics = run.summary

            # Process accuracy metrics
            val_metrics = {k: v for k, v in last_metrics.items() 
                         if k.startswith("Offline val. acc.") and v is not None}

            # Process R2 metrics
            val_r2_metrics = {}
            has_sklearn = False
            for k, v in last_metrics.items():
                if k.startswith("Offline sklearn regression test R2") and v is not None:
                    # Convert to new naming convention
                    new_key = k.replace("Offline sklearn regression test R2", "Offline val. R2")
                    val_r2_metrics[new_key] = v
                    has_sklearn = True
                elif k.startswith("Offline sklearn regression dense test R2") and v is not None:
                    # Convert to new naming convention
                    new_key = k.replace("Offline sklearn regression dense test R2", "Offline val. R2")
                    val_r2_metrics[new_key] = v
                    has_sklearn = True
                elif not has_sklearn and k.startswith("Offline val. R2") and v is not None:
                    # Keep existing R2 metrics
                    val_r2_metrics[k] = v

            # Increment counters
            if val_metrics:
                acc_metrics_found += 1
            if val_r2_metrics:
                r2_metrics_found += 1

            # Skip if no metrics found
            if not val_metrics and not val_r2_metrics:
                continue

            # get run config
            run_config = run.config
            # get number of occluded frames
            n_occluded_frames = run_config.get("spritevid_occlude_n_frames", 0)

            # get noise level
            if run_config.get("spritevid_noise_type", None) == "gaussian":
                noise_level = run_config.get("spritevid_noise_level", None)
            elif run_config.get("spritevid_noise_type", None) is None:
                noise_level = 0.0
            else:
                noise_level = None

            processed_runs += 1

            # Store accuracy metrics by task
            for key, value in val_metrics.items():
                task = key.split("(")[1].rstrip(")") if "(" in key else "default"
                if task not in results[matching_dataset][method]:
                    results[matching_dataset][method][task] = {'acc': [], 'n_occluded_frames': [], 'noise_level': []}
                results[matching_dataset][method][task]['acc'].append(value)
                results[matching_dataset][method][task]['n_occluded_frames'].append(n_occluded_frames)
                results[matching_dataset][method][task]['noise_level'].append(noise_level)

            # Store R2 metrics by task - ensure this is completely separate from accuracy processing
            for key, value in val_r2_metrics.items():
                task = key.split("(")[1].rstrip(")") if "(" in key else "default"
                # Store R2 values in the r2 section of the results dict
                if task not in results[matching_dataset][method]:
                    results[matching_dataset][method][task] = {'r2': [], 'n_occluded_frames': [], 'noise_level': []}
                if 'r2' not in results[matching_dataset][method][task]:
                    results[matching_dataset][method][task]['r2'] = []
                results[matching_dataset][method][task]['r2'].append(value)
                results[matching_dataset][method][task]['n_occluded_frames'].append(n_occluded_frames)
                results[matching_dataset][method][task]['noise_level'].append(noise_level)

            for task in results[matching_dataset][method]:
                # Ensure 'acc' and 'r2' keys exist
                if 'acc' not in results[matching_dataset][method][task]:
                    results[matching_dataset][method][task]['acc'] = []
                if 'r2' not in results[matching_dataset][method][task]:
                    results[matching_dataset][method][task]['r2'] = []
                # Ensure 'n_occluded_frames' and 'noise_level' are lists
                if 'n_occluded_frames' not in results[matching_dataset][method][task]:
                    results[matching_dataset][method][task]['n_occluded_frames'] = []
                if 'noise_level' not in results[matching_dataset][method][task]:
                    results[matching_dataset][method][task]['noise_level'] = []

        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
            continue

    print(f"Processed {processed_runs} runs")
    print(f"Found accuracy metrics in {acc_metrics_found} runs")
    print(f"Found R2 metrics in {r2_metrics_found} runs")

    return results


def results_to_dataframe(results):
    """Convert dictionary of loaded results from WandB to pandas dataframe."""
    # Create lists to store the data
    rows = []

    # Iterate through the nested dictionary
    for dataset in results:
        for method in results[dataset]:
            for task in results[dataset][method]:
                # Skip empty lists
                if not results[dataset][method][task]['acc'] and not results[dataset][method][task]['r2']:
                    continue

                # Get all values for this combination
                values = results[dataset][method][task]

                # Create a row for each seed
                for seed in range(max(len(values['r2']), len(values['acc']))):
                    # Create a row with all relevant information
                    row = {
                        'dataset': dataset,
                        'method': method,
                        'task': task,
                        'seed': seed,
                        'n_occluded_frames': values['n_occluded_frames'][seed],
                        'noise_level': values['noise_level'][seed],
                        'accuracy': values['acc'][seed] if values['acc'] != [] else None,
                        'r2': values['r2'][seed] if values['r2'] != [] else None
                    }

                    rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df


def combine_libri_datasets(results):
    """
    Combines specific tasks from libri datasets into new datasets:
    - libri_ctx: combines seq2seq from libri_delta8 and seq2label from libri_delta8_speaker_ctx
    - libri_enc: combines seq2seq from libri_delta8_phoneme_enc and seq2label from libri_delta8_speaker_enc
    
    Args:
        results: The nested dictionary from extract_all_results
    
    Returns:
        Updated results dictionary with the new combined datasets
    """
    # Create new dataset entries in results
    for dataset_name in ['libri_ctx', 'libri_enc']:
        results[dataset_name] = {
            'acc': {method: {} for method in results['libri_delta8']['acc'].keys()},
            'r2': {method: {} for method in results['libri_delta8']['r2'].keys()}
        }
    
    # Combine tasks for libri_ctx
    for metric_type in ['acc', 'r2']:
        for method in results['libri_delta8'][metric_type].keys():
            # Copy seq2seq task from libri_delta8 to libri_ctx if it exists
            if 'seq2seq' in results['libri_delta8'][metric_type][method]:
                results['libri_ctx'][metric_type][method]['seq2seq'] = \
                    results['libri_delta8'][metric_type][method]['seq2seq'].copy()
            
            # Copy seq2label task from libri_delta8_speaker_ctx to libri_ctx if it exists
            if 'seq2label' in results['libri_delta8_speaker_ctx'][metric_type][method]:
                results['libri_ctx'][metric_type][method]['seq2label'] = \
                    results['libri_delta8_speaker_ctx'][metric_type][method]['seq2label'].copy()
    
    # Combine tasks for libri_enc
    for metric_type in ['acc', 'r2']:
        for method in results['libri_delta8_phoneme_enc'][metric_type].keys():
            # Copy seq2seq task from libri_delta8_phoneme_enc to libri_enc if it exists
            if 'seq2seq' in results['libri_delta8_phoneme_enc'][metric_type][method]:
                results['libri_enc'][metric_type][method]['seq2seq'] = \
                    results['libri_delta8_phoneme_enc'][metric_type][method]['seq2seq'].copy()
            
            # Copy seq2label task from libri_delta8_speaker_enc to libri_enc if it exists
            if 'seq2label' in results['libri_delta8_speaker_enc'][metric_type][method]:
                results['libri_enc'][metric_type][method]['seq2label'] = \
                    results['libri_delta8_speaker_enc'][metric_type][method]['seq2label'].copy()
    
    return results


def generate_strictly_uniform_ticks(y_min, y_max, max_ticks=5, metric_type='accuracy'):
    """
    Generate strictly uniform ticks with guarantees of even spacing.
    
    Args:
        y_min (float): Lower y-axis limit
        y_max (float): Upper y-axis limit
        max_ticks (int): Maximum number of ticks to generate
        metric_type (str): Type of metric ('accuracy', 'r2', or other)
        
    Returns:
        numpy.ndarray: Array of uniformly spaced tick positions
    """
    import numpy as np
    
    # For accuracy or r2 metrics, cap at 1.0
    if metric_type in ('accuracy', 'r2'):
        effective_max = min(y_max, 1.0)
    else:
        effective_max = y_max
    
    # For very small ranges, just use min and max
    range_val = effective_max - y_min
    if range_val < 0.05:
        return np.array([y_min, effective_max])
    
    # Candidate step sizes in ascending order
    steps = np.array([0.025, 0.05, 0.1, 0.2, 0.25, 0.5])
    
    # Find appropriate step size
    step = steps[0]
    for candidate_step in steps:
        num_ticks = int(np.ceil(range_val / candidate_step)) + 1
        if num_ticks <= max_ticks:
            step = candidate_step
            break
    
    # If no step produces few enough ticks, use the largest
    if step == steps[0] and int(np.ceil(range_val / step)) + 1 > max_ticks:
        step = steps[-1]
    
    # Find starting tick: round y_min down to nearest multiple of step
    start = np.floor(y_min / step) * step
    
    # Ensure start is no more than one step below y_min
    if start < y_min - step/2:
        start += step
    
    # Generate ticks
    ticks = np.arange(start, effective_max + 0.0001, step)
    
    # Ensure no ticks exceed effective_max
    ticks = ticks[ticks <= effective_max + 0.0001]
    
    # Special handling for values close to 1.0 for accuracy/r2
    if metric_type in ('accuracy', 'r2'):
        # If close to 1.0 but not included, add it
        if y_max >= 0.95 and (len(ticks) == 0 or ticks[-1] < 0.999):
            if len(ticks) < max_ticks:
                ticks = np.append(ticks, 1.0)
            else:
                # Replace last tick with 1.0
                ticks[-1] = 1.0
    
    # Ensure we don't exceed max_ticks
    if len(ticks) > max_ticks:
        stride = int(np.ceil(len(ticks) / max_ticks))
        new_ticks = ticks[::stride]
        
        # Always include the last tick
        if new_ticks[-1] != ticks[-1]:
            if len(new_ticks) < max_ticks:
                new_ticks = np.append(new_ticks, ticks[-1])
            else:
                new_ticks[-1] = ticks[-1]
        
        ticks = new_ticks
    
    return ticks


def add_string_formatters_to_axes(ax):
    """Add custom formatters to prevent LaTeX rendering of numerical tick labels."""
    # Define formatter functions that convert numbers to strings
    def y_fmt(x, pos):
        return f"{x:.1f}"
    
    def x_fmt(x, pos):
        return f"{x:.1f}"
    
    # Apply the formatters to the axes
    ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    if len(ax.get_xticks()) > 0:  # Only if there are x-ticks
        ax.xaxis.set_major_formatter(FuncFormatter(x_fmt))


def performances_to_dataframe(method_performances, dataset_name, seed=0):
    """
    Convert performance dictionaries from compute_readout_projections into a DataFrame
    format expected by plot_advanced_comparison.
    
    Args:
        method_performances: Dictionary mapping method names to performance dictionaries 
                          from compute_readout_projections
        dataset_name: Name of the dataset
        seed: Seed value to use for all rows (default: 0)
        
    Returns:
        DataFrame with columns: dataset, method, task, seed, accuracy, r2
    """
    rows = []
    
    for method, perf in method_performances.items():
        # Process classification tasks (indices 0 and 1)
        for task_type_idx in [0, 1]:
            if task_type_idx < len(perf):
                for task_name, accuracy in perf[task_type_idx].items():
                    rows.append({
                        'dataset': dataset_name,
                        'method': method,
                        'task': task_name,
                        'seed': seed,
                        'accuracy': accuracy,
                        'r2': None
                    })
        
        # Process regression tasks (indices 2 and 3)
        for task_type_idx in [2, 3]:
            if task_type_idx < len(perf):
                for task_name, r2 in perf[task_type_idx].items():
                    rows.append({
                        'dataset': dataset_name,
                        'method': method,
                        'task': task_name,
                        'seed': seed,
                        'accuracy': None,
                        'r2': r2
                    })
        
        # Process special orientation tasks (index 4)
        if 4 < len(perf):
            for task_name, value in perf[4].items():
                if task_name == 'theta_discr':
                    # This is an accuracy task
                    rows.append({
                        'dataset': dataset_name,
                        'method': method,
                        'task': 'orientation_corrected',
                        'seed': seed,
                        'accuracy': value,
                        'r2': None
                    })
                else:
                    # These are regression tasks (sin, cos)
                    rows.append({
                        'dataset': dataset_name,
                        'method': method,
                        'task': task_name + '_corrected',
                        'seed': seed,
                        'accuracy': None,
                        'r2': value
                    })
    
    return pd.DataFrame(rows)


def sample_points_by_density(data, labels, n_samples=5000):
    """
    Sample points uniformly across the data space.
    
    Args:
        data: Data array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples, n_label_dims)
        n_samples: Number of points to sample
        
    Returns:
        Tuple of (sampled_data, sampled_labels)
    """
    # If we have fewer points than requested, return all
    if data.shape[0] <= n_samples:
        return data, labels
    
    # Sample indices uniformly
    indices = np.random.choice(data.shape[0], n_samples, replace=False)
    return data[indices], labels[indices]


def visualize_projections(
    projections_data,
    labels_data, 
    model_name=None,
    label_names=None,
    true_var_names=None,
    save_path=None,
    n_samples=5000,
    colormap='plasma',
    ignore_limits=False,
    show_xlabels=False,
    show_colorbar=True,
    figsize=(16, 20),
    simple_mode=False,
    use_corrected_orientation=True,
    plot_indices=None,
    xlim=None,
    ylim=None,
    scale_bar=False,
):
    """
    Unified function to visualize paired variable projections.
    
    Args:
        projections_data: List of dictionaries containing projection data
        labels_data: List of dictionaries containing label data
        model_name: Name of the model (for title)
        label_names: List of names for the label dimensions
        true_var_names: List of names for colorbar labels
        save_path: Path to save the figure (if None, figure is displayed)
        n_samples: Number of samples to plot
        colormap: Colormap to use for scatter plots
        ignore_limits: If True, use adaptive axis limits
        show_xlabels: If True, display x-axis labels on plots
        show_colorbar: If True, display colorbars for plots
        figsize: Figure size (width, height) in inches
        simple_mode: If True, use simplified layout with a single row
        use_corrected_orientation: If True, use orientation correction data (5th element),
                                  otherwise use orientation from standard data (4th element)
        plot_indices: List of indices to plot in simple mode (e.g., [0, 1, 4, 5] to plot
                     only specific plots out of the 8 possible plots)
    
    Returns:
        Figure handle
    """
    # Default parameter handling
    if label_names is None:
        label_names = true_var_names
    
    # Define the paired variables to visualize
    paired_vars = [
        (0, 1, 'Position (x, y)'),       # x-pos, y-pos
        (3, 4, 'Velocity (x, y)'),       # x-vel, y-vel
        (2, 5, 'Position/Velocity (z)'), # z-pos, z-vel
        (6, 7, 'Orientation (sin, cos)') # sin, cos
    ]
    
    # Extract standard regression data from the 4th element (index 3)
    std_data = {}
    std_labels = {}
    if len(projections_data) > 3 and len(labels_data) > 3:
        std_data = projections_data[3]  # 4th element
        std_labels = labels_data[3]     # 4th element
    else:
        raise ValueError("Standard regression data (4th element) not found in inputs")
    
    # Extract orientation correction data from the 5th element (index 4)
    orient_data = {}
    orient_labels = {}
    if len(projections_data) > 4 and len(labels_data) > 4 and use_corrected_orientation:
        orient_data = projections_data[4]  # 5th element
        orient_labels = labels_data[4]     # 5th element
    
    # Create unified projection array using standard and orientation data
    num_samples = 0
    
    # Check which features are available in the standard data to determine sample count
    std_features = {
        'x-position': 0,
        'y-position': 1,
        'z-position': 2,
        'x-velocity': 3,
        'y-velocity': 4,
        'z-velocity': 5
    }

    # Determine the number of samples from standard data
    for feature_name in std_features:
        if feature_name in std_labels and len(std_labels[feature_name]) > 0:
            num_samples = std_labels[feature_name].shape[0]
            break
    
    if num_samples == 0:
        raise ValueError("No valid standard data found in the provided inputs")
    
    # Initialize combined arrays with NaNs
    combined_projections = np.full((num_samples, 8), np.nan)
    combined_labels = np.full((num_samples, 8), np.nan)
    
    # Populate standard projection features (first 6 dimensions)
    for feature_name, idx in std_features.items():
        if feature_name in std_data and len(std_data[feature_name]) > 0:
            combined_projections[:, idx] = std_data[feature_name].flatten()
        
        if feature_name in std_labels and len(std_labels[feature_name]) > 0:
            combined_labels[:, idx] = std_labels[feature_name].flatten()
    
    # Handle orientation features (last 2 dimensions)
    if use_corrected_orientation and orient_data and 'sin' in orient_data and 'cos' in orient_data:
        # Use corrected orientation data from 5th element
        if 'sin' in orient_data and len(orient_data['sin']) > 0:
            combined_projections[:, 6] = orient_data['sin'].flatten()
        if 'cos' in orient_data and len(orient_data['cos']) > 0:
            combined_projections[:, 7] = orient_data['cos'].flatten()
        
        if 'sin' in orient_labels and len(orient_labels['sin']) > 0:
            combined_labels[:, 6] = orient_labels['sin'].flatten()
        if 'cos' in orient_labels and len(orient_labels['cos']) > 0:
            combined_labels[:, 7] = orient_labels['cos'].flatten()
    else:
        # Use orientation data from standard data (4th element)
        orient_std_features = {'sin': 6, 'cos': 7}
        for feature_name, idx in orient_std_features.items():
            if feature_name in std_data and len(std_data[feature_name]) > 0:
                combined_projections[:, idx] = std_data[feature_name].flatten()
            
            if feature_name in std_labels and len(std_labels[feature_name]) > 0:
                combined_labels[:, idx] = std_labels[feature_name].flatten()
    
    # Sample points for visualization
    proj_sample, labels_sample = sample_points_by_density(
        combined_projections, combined_labels, n_samples
    )
    
    # Determine which plots to display in simple mode
    if plot_indices is None:
        # Default to all 8 plots (0-7)
        plot_indices = list(range(8))
    
    # Set up the figure layout based on mode
    if simple_mode:
        # Use a single row layout similar to visualize_regressor_projections_simple
        num_plots = len(plot_indices)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, num_plots, figure=fig, wspace=0.1, width_ratios=[1] * num_plots)
    else:
        # Use a grid layout similar to visualize_regressor_projections
        fig = plt.figure(figsize=figsize)
        if model_name:
            fig.suptitle(f'Projections for {model_name}', fontsize=16, y=0.98)
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.4)
    
    # Find global min/max for each dimension to ensure consistent scaling
    global_mins = np.nanmin(proj_sample, axis=0)
    global_maxs = np.nanmax(proj_sample, axis=0)
    
    # Define the axis color for all plots
    axis_color = '#808080'  # Medium gray
    
    # Create plots
    plot_idx = 0
    
    for i, (idx1, idx2, title) in enumerate(paired_vars):
        # Calculate pair_idx based on current pair and color variables
        for j, color_idx in enumerate([idx1, idx2]):
            pair_plot_idx = i * 2 + j
            
            # Skip if this plot is not in the requested plot_indices for simple mode
            if simple_mode and pair_plot_idx not in plot_indices:
                continue
            
            # Check if this pair has valid data
            if (np.all(np.isnan(proj_sample[:, idx1])) or 
                np.all(np.isnan(proj_sample[:, idx2])) or
                np.isnan(global_mins[idx1]) or np.isnan(global_maxs[idx1]) or
                np.isnan(global_mins[idx2]) or np.isnan(global_maxs[idx2])):
                
                # Create empty placeholder for missing data
                if simple_mode:
                    ax = fig.add_subplot(gs[0, plot_idx])
                else:
                    ax = fig.add_subplot(gs[i, j])
                
                ax.text(0.5, 0.5, 'Data N/A', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                if simple_mode:
                    plot_idx += 1
                continue
            
            # Calculate data ranges for this pair
            x_min, x_max = global_mins[idx1], global_maxs[idx1]
            y_min, y_max = global_mins[idx2], global_maxs[idx2]
            
            # Get subplot position based on mode
            if simple_mode:
                ax = fig.add_subplot(gs[0, plot_idx])
                plot_idx += 1
            else:
                ax = fig.add_subplot(gs[i, j])
            
            # Plot predictions colored by the selected variable
            scatter = ax.scatter(
                proj_sample[:, idx1],
                proj_sample[:, idx2],
                c=labels_sample[:, color_idx],
                cmap=colormap,
                alpha=0.8, 
                s=20 if not simple_mode else 1,
                edgecolor='none'
            )
            
            # Add colorbar if enabled
            if show_colorbar:
                if simple_mode:
                    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                                        aspect=10, pad=0.35)
                    cbar.outline.set_linewidth(0.5)
                    cbar.set_label(f'{true_var_names[color_idx]}', fontsize=8, labelpad=2)
                    
                    # Set appropriate ticks based on the pair
                    if i == 2 and j == 0:  # Z-pos colored by Z-pos
                        cbar.set_ticks([0, 1])
                    else:
                        cbar.set_ticks([-1, 0, 1])
                    
                    cbar.ax.tick_params(labelsize=6)
                else:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(f'{true_var_names[color_idx]} value')
            
            # Set axis labels and title
            if not simple_mode:
                ax.set_xlabel(f'Projection of {label_names[idx1]}')
                ax.set_ylabel(f'Projection of {label_names[idx2]}')
                
                # Add title with variable information
                plot_title = f'{title}\nColored by {label_names[color_idx]}'
                ax.set_title(plot_title, fontsize=11)
            else:
                # For simple mode, remove all axis labels as requested
                ax.set_xlabel('')
                ax.set_ylabel('')
            
            # Set axis limits using the original methods
            if simple_mode:
                # Original calculation from visualize_regressor_projections_simple
                if i < 3:  # Standard regression pairs
                    x_abs_max = max(abs(x_min), abs(x_max))
                    y_abs_max = max(abs(y_min), abs(y_max))
                    max_abs = max(x_abs_max, y_abs_max) * 1.1
                else:  # Orientation pair
                    x_abs_max = max(abs(x_min), abs(x_max)) if not (np.isnan(x_min) or np.isnan(x_max)) else 1.0
                    y_abs_max = max(abs(y_min), abs(y_max)) if not (np.isnan(y_min) or np.isnan(y_max)) else 1.0
                    max_abs = max(x_abs_max, y_abs_max) * 1.1
                    if max_abs == 0:
                        max_abs = 1.0
                
                if xlim is not None:
                    ax.set_xlim(xlim)
                    max_abs = max(xlim)
                else:
                    ax.set_xlim(-max_abs, max_abs)
                if ylim is not None:
                    ax.set_ylim(ylim)
                else:
                    ax.set_ylim(-max_abs, max_abs)
                # Set equal aspect ratio
                ax.set_aspect('equal')
                
                # Remove all axes elements for simple mode
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                # Add indicator for the origin
                ax.plot([0], [0], '+', color='black', markersize=5, linewidth=0.5)

                # # Add scale bars
                if scale_bar:
                    bar_length = 1.0  # Length of the scale bar in data units
                    ax.hlines(y=-max_abs - 0.02 * max_abs, xmin=0,
                              xmax=bar_length, color='black', linewidth=4,
                              zorder=5)

            else:
                # For non-simple mode, enforce square aspect ratio
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Calculate bounds to ensure square aspect ratio
                max_range = max(x_range, y_range) * 1.1
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                
                # Set limits to create square aspect ratio
                ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
                ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
                
                if not ignore_limits:
                    if i == 2:  # Z-pos/Z-vel pair
                        ax.set_xticks([0, 2])
                    else:
                        ax.set_xticks([-1, 1])
                    ax.set_yticks([-1, 1])
                else:
                    # Use adaptive tick positions based on ceiling of max_abs
                    max_abs = max_range/2
                    x_tick_neg = -np.ceil(max_abs)
                    x_tick_pos = np.ceil(max_abs)
                    y_tick_neg = -np.ceil(max_abs)
                    y_tick_pos = np.ceil(max_abs)
                    
                    # Handle cases where max_abs is very small or zero
                    if x_tick_neg == x_tick_pos:  # handles max_abs near 0
                        x_ticks = np.array([x_tick_neg - 1, x_tick_neg, x_tick_neg + 1]) if x_tick_neg == 0 else np.array([x_tick_neg, 0, -x_tick_neg]) if x_tick_neg < 0 else np.array([-x_tick_pos, 0, x_tick_pos])
                    else:
                        x_ticks = [x_tick_neg, x_tick_pos]

                    if y_tick_neg == y_tick_pos:
                        y_ticks = np.array([y_tick_neg - 1, y_tick_neg, y_tick_neg + 1]) if y_tick_neg == 0 else np.array([y_tick_neg, 0, -y_tick_neg]) if y_tick_neg < 0 else np.array([-y_tick_pos, 0, y_tick_pos])
                    else:
                        y_ticks = [y_tick_neg, y_tick_pos]
                    
                    ax.set_xticks(sorted(list(set(x_ticks))))  # Use unique sorted ticks
                    ax.set_yticks(sorted(list(set(y_ticks))))
            
            # Set equal aspect ratio explicitly
            ax.set_aspect('equal')
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    
    return fig


def visualize_3d_projections(
    projections_data,
    labels_data,
    model_name=None,
    save_path=None,
    n_samples=5000,
    elevation=30,
    azimuth=45,
    figsize=(15, 5),
    class_colors=None,
    paired_vars=None,
    plot_indices=None,
    point_size=2,
    point_alpha=0.2,
    z_amp_factor=1.0,
):
    """
    Visualize 3D projections with binary classifier as the third dimension,
    styled consistently with visualize_projections simple_mode.
    
    Args:
        projections_data: List of dictionaries containing projection data
        labels_data: List of dictionaries containing label data
        model_name: Name of the model (for title)
        save_path: Path to save the figure
        n_samples: Number of samples to plot
        colormap: Colormap to use for scatter plots
        elevation: 3D plot elevation angle
        azimuth: 3D plot azimuth angle
        figsize: Figure size (width, height) in inches
        class_colors: Dictionary mapping class labels to colors
        paired_vars: List of paired variables to visualize
        plot_indices: List of indices to plot
        scale_bar: Whether to show scale bars (length 1) on plots
        point_size: Size of scatter plot points
        point_alpha: Alpha/transparency of points
        class_idxes: List of two class indices to classify (e.g., [0, 7])
    
    Returns:
        Figure handle
    """
    # Default parameters
    class_idxes = [0, 1]
    if class_colors is None:
        class_colors = {
            class_idxes[0]: '#44AA99',  # Teal for first class
            class_idxes[1]: '#AA4499'   # Purple for second class
        }
    
    if paired_vars is None:
        # Default to velocity and orientation pairs
        paired_vars = [
            (3, 4, r'$\hat{v}_x$', r'$\hat{v}_y$', r'$\hat{S}$'),      # x-vel, y-vel, classifier
            (6, 7, r'$\hat{\sin\theta}$', r'$\hat{\cos\theta}$', r'$\hat{S}$')    # sin, cos, classifier
        ]
    
    # Determine which plots to display
    if plot_indices is None:
        # Default to all plots
        plot_indices = list(range(len(paired_vars)))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    if model_name:
        fig.suptitle(f'3D Projections for {model_name}', fontsize=10, y=0.95)
    
    num_plots = len(plot_indices)
    gs = GridSpec(1, num_plots, figure=fig, wspace=-0.25, width_ratios=[1] * num_plots)
    
    # Extract binary classifier data from projections_data
    # The binary classifier should be in index 5 (6th element) based on your structure
    if len(projections_data) > 5 and 'bin' in projections_data[5]:
        clf_projections = projections_data[5]['bin']  # Binary classifier logits
        # Get the corresponding labels - we need sprite labels to determine which samples belong to our classes
        
        # Extract sprite labels from your data structure
        # We need to get the sprite labels that correspond to the binary classifier
        if len(labels_data) > 5 and 'bin' in labels_data[5]:
            # These are the binary labels (0, 1) for the two classes
            binary_labels = labels_data[5]['bin']
            
            # For now, let's assume we can reconstruct which samples belong to which original class
            # We'll create a mask for the classes we're interested in
            mask = np.isin(binary_labels, class_idxes)
            clf_projections = clf_projections[mask]
            binary_labels = binary_labels[mask]
            
        else:
            raise ValueError("Binary classifier labels not found in labels_data[5]['bin']")
    else:
        raise ValueError("Binary classifier projections not found in projections_data[5]['bin']")
    
    # Get the standard regression projections for x, y coordinates
    if len(projections_data) > 3:
        std_projections = projections_data[3]  # Standard regression data
    else:
        raise ValueError("Standard regression projections not found")
    
    # Build the combined projection array
    # We need to match the samples between binary classifier and standard projections
    n_samples_total = len(clf_projections)
    
    # Initialize combined projections array
    combined_projections = np.full((n_samples_total, 8), np.nan)
    
    # Map standard regression features
    feature_mapping = {
        'x-position': 0, 'y-position': 1, 'z-position': 2,
        'x-velocity': 3, 'y-velocity': 4, 'z-velocity': 5,
        'sin': 6, 'cos': 7
    }
    
    # Fill in the standard projections
    for feature_name, idx in feature_mapping.items():
        if feature_name in std_projections and len(std_projections[feature_name]) > 0:
            subsampled_projections = std_projections[feature_name][mask]
            combined_projections[:, idx] = subsampled_projections.flatten()
    
    # Use corrected orientation if available (index 4)
    if len(projections_data) > 4:
        orient_projections = projections_data[4]
        if 'sin' in orient_projections:
            combined_projections[:, 6] = orient_projections['sin'][mask.squeeze()].flatten()
        if 'cos' in orient_projections:
            combined_projections[:, 7] = orient_projections['cos'][mask.squeeze()].flatten()
    
    # Sample points for visualization while preserving class balance
    if n_samples_total > n_samples:
        # Sample points while trying to maintain class balance
        indices = np.arange(n_samples_total)
        sampled_indices = np.random.choice(indices, n_samples, replace=False)
        
        proj_sample = combined_projections[sampled_indices]
        clf_sample = clf_projections[sampled_indices]
        class_sample = binary_labels[sampled_indices]
    else:
        proj_sample = combined_projections
        clf_sample = clf_projections * z_amp_factor
        class_sample = binary_labels
    
    # Create plots
    for plot_idx, pair_idx in enumerate(plot_indices):
        if pair_idx >= len(paired_vars):
            continue
            
        idx1, idx2, x_label, y_label, z_label = paired_vars[pair_idx]
        
        # Check if this pair has valid data
        if (np.all(np.isnan(proj_sample[:, idx1])) or 
            np.all(np.isnan(proj_sample[:, idx2])) or
            clf_sample is None):
            
            # Create empty placeholder for missing data
            ax = fig.add_subplot(gs[0, plot_idx])
            
            ax.text(0.5, 0.5, 'Data N/A', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            # make transparent background
            ax.set_facecolor((1.0, 1.0, 1.0, 0.0))

            # reduce margins
            ax.margins(0.1)
            
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            continue
        
        # Create 3D subplot
        ax = fig.add_subplot(gs[0, plot_idx], projection='3d')
        
        # Calculate data ranges for this pair
        x_min, x_max = np.nanmin(proj_sample[:, idx1]), np.nanmax(proj_sample[:, idx1])
        y_min, y_max = np.nanmin(proj_sample[:, idx2]), np.nanmax(proj_sample[:, idx2])
        z_min, z_max = np.nanmin(clf_sample), np.nanmax(clf_sample)
        
        # Make limits symmetric
        x_abs_max = max(abs(x_min), abs(x_max)) * 1.1
        y_abs_max = max(abs(y_min), abs(y_max)) * 1.1
        z_abs_max = max(abs(z_min), abs(z_max)) * 1.1
        
        # Plot points by class
        for class_id, class_color in class_colors.items():
            class_mask = class_sample == class_id
            
            if np.sum(class_mask) > 0:
                # Create color array with consistent alpha
                rgba_color = np.zeros((np.sum(class_mask), 4))
                rgb_color = to_rgb(class_color)
                rgba_color[:, 0] = rgb_color[0]
                rgba_color[:, 1] = rgb_color[1]
                rgba_color[:, 2] = rgb_color[2]
                rgba_color[:, 3] = point_alpha
                
                ax.scatter(
                    proj_sample[class_mask, idx1],
                    proj_sample[class_mask, idx2],
                    clf_sample[class_mask].flatten(),
                    c=rgba_color,
                    s=point_size,
                    edgecolor='none'
                )
        
        # Set view angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        ax.set_xlim(-x_abs_max, x_abs_max)
        ax.set_ylim(-y_abs_max, y_abs_max)
        ax.set_zlim(-z_abs_max, z_abs_max)
        
        # Remove rotation for x and y labels to match 2D simple mode
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)
        
        # Style the plot for a clean, minimal look like simple_mode
        # Make panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Remove grid
        ax.grid(False)

        # Turn off all axes lines
        ax.xaxis.line.set_color('none')
        ax.yaxis.line.set_color('none')
        ax.zaxis.line.set_color('none')
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.show()

    # Save figure if requested
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    
    return fig


def plot_dataset_progression(results_df, methods_to_plot, baseline_methods, metric_configs, 
                           method_names, method_colors=None, figsize=None, ylim=None, 
                           loc_legend='best', display_legend=True, legend_ax_idx=-1, 
                           plot_spacing=0.3, legend_ncols=1, point_size=25, 
                           line_alpha=0.7, error_bars=True, xlabel='Dataset Number',
                           xticklabels=None):
    """
    Plot performance metrics as a function of dataset number progression.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with columns ['dataset', 'method', 'task', 'seed', 'accuracy', 'r2']
    methods_to_plot : list
        List of method names to plot as curves
    baseline_methods : list
        List of baseline method names to plot as horizontal lines
    metric_configs : dict
        Configuration for metrics and tasks, same format as plot_advanced_comparison
    method_names : dict
        Mapping from method keys to display names
    method_colors : dict, optional
        Mapping from method keys to colors
    figsize : tuple, optional
        Figure size (width, height)
    ylim : dict, optional
        Y-limits for each metric type, e.g., {'accuracy': (0.5, 1.0)}
    loc_legend : str
        Legend location
    display_legend : bool
        Whether to display legend
    legend_ax_idx : int
        Which subplot to put legend on (-1 for last)
    plot_spacing : float
        Spacing between subplots
    legend_ncols : int
        Number of columns in legend
    point_size : int
        Size of scatter points
    line_alpha : float
        Alpha for line plots
    error_bars : bool
        Whether to show error bars (std dev)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    
    # Extract dataset numbers and sort
    dataset_numbers = []
    dataset_base_name = None
    
    for dataset in results_df['dataset'].unique():
        # Extract the number at the end
        import re
        match = re.match(r'(.+?)(\d+)$', dataset)
        if match:
            base_name, number = match.groups()
            if dataset_base_name is None:
                dataset_base_name = base_name
            elif dataset_base_name != base_name:
                raise ValueError(f"Inconsistent dataset base names: {dataset_base_name} vs {base_name}")
            dataset_numbers.append(int(number))
    
    dataset_numbers = sorted(set(dataset_numbers))
    
    if not dataset_numbers:
        raise ValueError("No valid dataset numbers found in dataset names")
    
    # Calculate subplot layout
    metric_types = list(metric_configs.keys())
    num_metrics = len(metric_types)
    
    # Calculate total number of task subplots
    total_subplots = 0
    metric_subplot_counts = {}
    
    for metric_type, config in metric_configs.items():
        individual_count = len(config.get('individual_tasks', {}))
        averaged_count = len(config.get('averaged_tasks', []))
        subplot_count = individual_count + averaged_count
        metric_subplot_counts[metric_type] = subplot_count
        total_subplots += subplot_count
    
    # Calculate figure size if not provided
    if figsize is None:
        width = min(16, max(4, 3 * total_subplots))
        height = 4
        figsize = (width, height)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Calculate width ratios based on number of tasks per metric
    width_ratios = []
    for metric_type in metric_types:
        count = metric_subplot_counts[metric_type]
        if count > 0:
            width_ratios.extend([1] * count)
    
    # Create GridSpec
    gs = gridspec.GridSpec(1, total_subplots, width_ratios=width_ratios, wspace=plot_spacing)
    
    axes = []
    all_legend_elements = []
    subplot_idx = 0
    
    # Process each metric type
    for metric_type in metric_types:
        config = metric_configs[metric_type]
        
        # Collect all task configurations for this metric
        task_configs = []
        
        # Process individual tasks
        for task, display_name in config.get('individual_tasks', {}).items():
            task_configs.append({
                'task_id': task,
                'display_name': display_name,
                'type': 'individual'
            })
        
        # Process averaged tasks
        for i, avg_config in enumerate(config.get('averaged_tasks', [])):
            task_configs.append({
                'task_id': f'avg_{i}',
                'display_name': avg_config['name'],
                'type': 'averaged',
                'tasks_to_average': avg_config['tasks']
            })
        
        # Create subplot for each task in this metric
        for task_config in task_configs:
            ax = fig.add_subplot(gs[0, subplot_idx])
            axes.append(ax)
            
            # Prepare data for this task across all datasets
            if task_config['type'] == 'individual':
                # Single task data
                task_data = results_df[results_df['task'] == task_config['task_id']].copy()
            else:
                # Averaged task data
                tasks_to_avg = task_config['tasks_to_average']
                avg_data_list = []
                
                for dataset_num in dataset_numbers:
                    dataset_name = f"{dataset_base_name}{dataset_num}"
                    dataset_subset = results_df[results_df['dataset'] == dataset_name]
                    
                    for method in methods_to_plot + baseline_methods:
                        method_subset = dataset_subset[
                            (dataset_subset['method'] == method) & 
                            (dataset_subset['task'].isin(tasks_to_avg))
                        ]
                        
                        if not method_subset.empty:
                            # Average across tasks for each seed
                            avg_by_seed = method_subset.groupby('seed')[metric_type].mean().reset_index()
                            avg_by_seed['dataset'] = dataset_name
                            avg_by_seed['method'] = method
                            avg_by_seed['task'] = task_config['task_id']
                            avg_data_list.append(avg_by_seed)
                
                if avg_data_list:
                    task_data = pd.concat(avg_data_list, ignore_index=True)
                else:
                    task_data = pd.DataFrame()
            
            # Plot method curves
            legend_elements = []
            
            for method in methods_to_plot:
                method_data = task_data[task_data['method'] == method]
                
                if not method_data.empty:
                    # Calculate mean and std for each dataset number
                    x_vals = []
                    y_means = []
                    y_stds = []
                    
                    for dataset_num in dataset_numbers:
                        dataset_name = f"{dataset_base_name}{dataset_num}"
                        dataset_method_data = method_data[method_data['dataset'] == dataset_name]
                        
                        if not dataset_method_data.empty:
                            values = dataset_method_data[metric_type].values
                            x_vals.append(dataset_num)
                            y_means.append(np.mean(values))
                            y_stds.append(np.std(values))
                    
                    if x_vals:
                        color = method_colors.get(method, f'C{methods_to_plot.index(method)}') if method_colors else f'C{methods_to_plot.index(method)}'
                        label = method_names.get(method, method)
                        
                        # Plot main line
                        line = ax.plot(x_vals, y_means, color=color, label=label, 
                                     linewidth=2, alpha=line_alpha, marker='o', markersize=4)[0]
                        
                        # Add error bars if requested
                        if error_bars and len(y_stds) > 0:
                            ax.fill_between(x_vals, 
                                          np.array(y_means) - np.array(y_stds),
                                          np.array(y_means) + np.array(y_stds),
                                          color=color, alpha=0.2)
                        
                        # Save legend element for last subplot
                        if subplot_idx == total_subplots - 1:
                            legend_elements.append(line)
            
            # Plot baseline horizontal lines
            for baseline_method in baseline_methods:
                baseline_data = task_data[task_data['method'] == baseline_method]
                
                if not baseline_data.empty:
                    # Calculate overall baseline value (average across all datasets)
                    baseline_values = []
                    for dataset_num in dataset_numbers:
                        dataset_name = f"{dataset_base_name}{dataset_num}"
                        dataset_baseline_data = baseline_data[baseline_data['dataset'] == dataset_name]
                        if not dataset_baseline_data.empty:
                            baseline_values.extend(dataset_baseline_data[metric_type].values)
                    
                    if baseline_values:
                        baseline_mean = np.mean(baseline_values)
                        
                        if baseline_method in baseline_styles:
                            color, style = baseline_styles[baseline_method]
                            label = method_names.get(baseline_method, baseline_method)
                            
                            line = ax.axhline(y=baseline_mean, color=color, linestyle=style, 
                                            label=label if subplot_idx == 0 else None,
                                            linewidth=1, alpha=0.8)
                            
                            # Save legend element for last subplot
                            if subplot_idx == total_subplots - 1:
                                legend_elements.append(line)
            
            # Set labels and limits
            ax.set_title(task_config['display_name'], fontsize=10)
            ax.set_xlabel(xlabel, fontsize=9)
            
            if subplot_idx == 0:  # Only first subplot gets y-label
                if metric_type == 'accuracy':
                    ax.set_ylabel('Accuracy', fontsize=9)
                elif metric_type == 'r2':
                    ax.set_ylabel(r'$R^2$', fontsize=9)
                else:
                    ax.set_ylabel(metric_type.upper(), fontsize=9)
            
            # Set y-limits if provided
            if ylim and metric_type in ylim:
                ax.set_ylim(ylim[metric_type])
            
            # Set x-limits
            if dataset_numbers:
                x_padding = (max(dataset_numbers) - min(dataset_numbers)) * 0.05
                ax.set_xlim(min(dataset_numbers) - x_padding, max(dataset_numbers) + x_padding)
            
            if xticklabels is not None:
                ax.set_xticks(dataset_numbers)
                ax.set_xticklabels(xticklabels, fontsize=8)
            
            # Store legend elements from last subplot
            if subplot_idx == total_subplots - 1:
                all_legend_elements = legend_elements
            
            subplot_idx += 1
    
    # Add legend
    if display_legend and all_legend_elements:
        # Determine which axis to put legend on
        if legend_ax_idx == -1 or legend_ax_idx >= len(axes):
            legend_ax = axes[-1]
        else:
            legend_ax = axes[legend_ax_idx]
        
        # Get labels
        labels = [elem.get_label() for elem in all_legend_elements]
        
        legend_ax.legend(all_legend_elements, labels,
                        loc=loc_legend,
                        fontsize=8,
                        frameon=False,
                        handlelength=1.5,
                        handletextpad=0.4,
                        labelspacing=0.4,
                        ncols=legend_ncols)
    
    # plt.tight_layout()
    return fig, axes


def plot_sin_cos_vs_theta(
    projections_data, 
    labels_data, 
    trig_function='sin', 
    model_name=None, 
    save_path=None, 
    n_samples=5000, 
    show_metrics=False, 
    prediction_color='#994455', 
    show_xlabel=True,
    show_ylabel=True,
    use_corrected_orientation=True
):
    """
    Create a scatter plot comparing predicted sine or cosine values against true theta (angle),
    with minimal styling. Adapted for compute_readout_projections output.
    
    Args:
        projections_data: List of dictionaries from compute_readout_projections
        labels_data: List of dictionaries from compute_readout_projections
        trig_function: Which trigonometric function to plot ('sin' or 'cos')
        model_name: Name of the model (optional)
        save_path: Path to save the figure (if None, figure is displayed)
        n_samples: Number of samples to plot
        show_metrics: Whether to show MSE and R² metrics in the title (default: False)
        prediction_color: Color for the predicted values
        show_xlabel: Whether to show x-axis label and ticks
        use_corrected_orientation: Whether to use corrected orientation (index 4) or standard (index 3)
    
    Returns:
        Tuple of (fig, ax)
    """
    # Check if trig_function is valid
    if trig_function not in ['sin', 'cos']:
        raise ValueError(f"trig_function must be 'sin' or 'cos', got '{trig_function}'")
    
    # Extract data based on use_corrected_orientation flag
    if use_corrected_orientation and len(projections_data) > 4:
        # Use corrected orientation data (index 4)
        orientation_projections = projections_data[4]
        orientation_labels = labels_data[4]
        
        if trig_function not in orientation_projections or trig_function not in orientation_labels:
            raise ValueError(f"Corrected orientation data missing {trig_function}")
        
        predicted_trig = orientation_projections[trig_function]
        true_trig = orientation_labels[trig_function]
        
        # For theta calculation, we need both sin and cos
        if 'sin' in orientation_labels and 'cos' in orientation_labels:
            true_sin = orientation_labels['sin']
            true_cos = orientation_labels['cos']
        else:
            raise ValueError("Both sin and cos needed for theta calculation in corrected orientation data")
    
    else:
        # Use standard regression data (index 3)
        if len(projections_data) <= 3:
            raise ValueError("Standard regression data not found")
        
        std_projections = projections_data[3]
        std_labels = labels_data[3]
        
        if trig_function not in std_projections or trig_function not in std_labels:
            raise ValueError(f"Standard regression data missing {trig_function}")
        
        predicted_trig = std_projections[trig_function]
        true_trig = std_labels[trig_function]
        
        # For theta calculation
        if 'sin' in std_labels and 'cos' in std_labels:
            true_sin = std_labels['sin']
            true_cos = std_labels['cos']
        else:
            raise ValueError("Both sin and cos needed for theta calculation in standard data")
    
    # Convert to numpy arrays and flatten
    predicted_trig = np.array(predicted_trig).flatten()
    true_trig = np.array(true_trig).flatten()
    true_sin = np.array(true_sin).flatten()
    true_cos = np.array(true_cos).flatten()
    
    # Sample points for visualization
    n_total = len(predicted_trig)
    if n_total > n_samples:
        indices = np.random.choice(n_total, n_samples, replace=False)
        predicted_trig_sample = predicted_trig[indices]
        true_trig_sample = true_trig[indices]
        true_sin_sample = true_sin[indices]
        true_cos_sample = true_cos[indices]
    else:
        predicted_trig_sample = predicted_trig
        true_trig_sample = true_trig
        true_sin_sample = true_sin
        true_cos_sample = true_cos
    
    # Get true angles (theta) using arctan2
    true_theta = np.arctan2(true_sin_sample, true_cos_sample)
    
    # Define the axis color
    axis_color = '#808080'
    
    # Create figure (using the original small size)
    fig = plt.figure(figsize=(0.7, 0.7))
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    
    # Create scatter plot with fixed color and true theta on x-axis
    scatter = ax.scatter(
        true_theta,
        predicted_trig_sample,
        c=prediction_color,
        alpha=0.8,
        s=1,
        edgecolor='none'
    )
    
    # Set axis limits
    ax.set_xlim(-np.pi-0.1, np.pi+0.1)  # x-axis (theta) from -π to π
    ax.set_ylim(-1.3, 1.3)      # y-axis (sin/cos) from -1.3 to 1.3
    
    # Set ticks
    ax.set_yticks([-1, 1])
    # ax.set_yticklabels([r"$-1$", r"$1$"], fontsize=8)
    if show_xlabel:
        ax.set_xticks([-np.pi, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$\pi$"], fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    
    # Simplify tick params
    ax.tick_params(axis='both', which='major', length=3, width=0.5, 
                  color=axis_color, pad=2, direction='in')    

    # Put xaxis in the middle
    ax.spines['bottom'].set_position('center')
    ax.spines['left'].set_position('zero')
    # Move the xticklabels to the bottom by increasing the bottom padding
    # Add reference curve - true sine or cosine function
    theta_range = np.linspace(-np.pi, np.pi, 1000)
    if trig_function == 'sin':
        reference = np.sin(theta_range)
    else:  # 'cos'
        reference = np.cos(theta_range)
    
    ax.plot(theta_range, reference, '-', color='black', linewidth=1.0, alpha=0.7)
    
    # Create LaTeX-formatted labels
    latex_names = {'sin': r'a', 'cos': r'b'}
    latex_var = latex_names[trig_function]
    
    # Create labels with LaTeX formatting
    if show_xlabel:
        ax.set_xlabel(r'$\theta$', fontsize=8, labelpad=2)
        ax.xaxis.set_tick_params(pad=3)
        # move the left-most xticklabel to the left
        leftmost_label = min(ax.get_xticklabels(), key=lambda x: x.get_position()[0])
        leftmost_label.set_x(-np.pi - 0.5)  # Adjust position to the left
        leftmost_label.set_ha('right')  # Align text to the right
        ax.xaxis.set_label_coords(1.1, 0.55)  # Adjust x-label position
    if show_ylabel:
        ax.set_ylabel(f'$\hat{{{latex_var}}}$', fontsize=8)
        ax.yaxis.set_tick_params(pad=3)

        # rotate the ylabel to be horizontal
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.set_label_coords(0.5, 1.05)  # Adjust y-label position

    # move the bottom-most yticklabel to the bottom
    ax.yaxis.set_tick_params(pad=1)
    bottommost_label = min(ax.get_yticklabels(), key=lambda x: x.get_position()[1])
    bottommost_label.set_verticalalignment('top')  # Align text to the bottom
    
    # Use seaborn to despine (remove top and right spines)
    sns.despine(fig=fig, ax=ax)
    
    # Calculate and add error metrics only if requested
    if show_metrics:
        # Calculate MSE and R² using the true trig values (not from theta)
        mse = np.mean((predicted_trig_sample - true_trig_sample) ** 2)
        
        # R² calculation
        ss_res = np.sum((true_trig_sample - predicted_trig_sample) ** 2)
        ss_tot = np.sum((true_trig_sample - np.mean(true_trig_sample)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        if model_name:
            title = f"{model_name}: {trig_function} vs theta\nMSE: {mse:.3f}, R²: {r_squared:.3f}"
        else:
            title = f"{trig_function} vs theta\nMSE: {mse:.3f}, R²: {r_squared:.3f}"
            
        ax.set_title(title, fontsize=10, pad=10)
    
    # Save or show figure
    if save_path:
        # Make sure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Save the figure
        fig.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close(fig)
    else:
        plt.show()
    
    return fig, ax


def plot_variable_vs_true(x_projections, labels, variable_name, model_name=None, 
                          save_path=None, n_samples=5000, show_metrics=False, 
                          show_xlabel=True, show_ylabel=True, color='#994455'):
    """
    Create a scatter plot comparing predicted variable values against true values,
    with minimal styling in the style of the angle_vs_true plot.
    
    Args:
        regressor_results: Dictionary containing regressor projections and labels
        variable_name: String name of the variable to plot ('sin', 'cos', 'x', 'y', 'vx', 'vy', 'z', 'vz', 'theta')
        model_name: Name of the model (optional)
        save_path: Path to save the figure (if None, figure is displayed)
        n_samples: Number of samples to plot
        show_metrics: Whether to show MSE and R² metrics in the title (default: False)
    
    Returns:
        Tuple of (fig, ax)
    """

    x_projections = np.array(x_projections)
    labels = np.array(labels)
    
    # Sample points for visualization
    if x_projections.shape[0] > n_samples:
        indices = np.random.choice(x_projections.shape[0], n_samples, replace=False)
        x_proj_sample = x_projections[indices]
        labels_sample = labels[indices]
    else:
        x_proj_sample = x_projections
        labels_sample = labels
    
    # Special case for theta (angle)
    if variable_name == 'theta':
        # Find the sine and cosine indices
        sine_index, cosine_index = 6, 7
        
        # Get predicted sine and cosine values
        predicted_sin = x_proj_sample[:, sine_index]
        predicted_cos = x_proj_sample[:, cosine_index]
        
        # Get true sine and cosine values
        true_sin = labels_sample[:, sine_index]
        true_cos = labels_sample[:, cosine_index]
        
        # Calculate angles using arctan2 (handles quadrant correctly)
        predicted_var = np.arctan2(predicted_sin, predicted_cos)
        true_var = np.arctan2(true_sin, true_cos)
    else:
        # Get predicted and true values for the selected variable
        predicted_var = x_proj_sample.squeeze()
        true_var = labels_sample.squeeze()
    
    # Define the axis color
    axis_color = '#808080'
    
    # Set fixed color for prediction
    prediction_color = color
    
    # Create figure (using the original small size)
    fig = plt.figure(figsize=(0.7, 0.7))
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    
    # Create scatter plot with fixed color
    scatter = ax.scatter(
        true_var,
        predicted_var,
        c=prediction_color,
        alpha=0.8,
        s=1,
        edgecolor='none'
    )
    
    # Set axis limits - determine a good range from the data
    min_val = min(np.min(true_var), np.min(predicted_var))
    max_val = max(np.max(true_var), np.max(predicted_var))
    padding = (max_val - min_val) * 0.05  # 5% padding
    
    # Special case for sin/cos: set fixed limits of [-1, 1]
    if variable_name in ['sin', 'cos']:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    
    # For theta (angle): set fixed limits of [-π, π]
    if variable_name == 'theta':
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, np.pi])
        ax.set_yticks([-np.pi, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$\pi$"], fontsize=8)
        ax.set_yticklabels([r"$-\pi$", r"$\pi$"], fontsize=8)
    else:
        ax.set_xlim(-1 - padding, 1 + padding)
        ax.set_ylim(-1 - padding, 1 + padding)
        min_true = np.min(true_var)
        max_true = np.max(true_var)
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_xticklabels([-1, 1], fontsize=8)
        ax.set_yticklabels([-1, 1], fontsize=8)
    
    # Simplify tick params
    ax.tick_params(axis='both', which='major', length=3, width=0.5, 
                  color=axis_color, pad=2, direction='in')
    
    # Add diagonal reference line
    ax.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 
           '-', color='black', linewidth=1.0, alpha=0.7)
    
    # Create LaTeX-formatted variable names for labels
    latex_names = {
        'x': r'r_1',
        'y': r'r_2',
        'z': r'r_3',
        'vx': r'v_1',
        'vy': r'v_2',
        'vz': r'v_3',
        'sin': r'\sin\theta',
        'cos': r'\cos\theta',
        'theta': r'\theta'
    }
    
    latex_var = latex_names.get(variable_name, variable_name)

    # Put xaxis in the middle
    ax.spines['bottom'].set_position('center')
    ax.spines['left'].set_position('zero')

    # Create labels with LaTeX formatting
    if show_xlabel:
        ax.set_xlabel(f'${latex_var}$', fontsize=8, labelpad=2)
        ax.xaxis.set_tick_params(pad=3)
        # move the left-most xticklabel to the left
        leftmost_label = min(ax.get_xticklabels(), key=lambda x: x.get_position()[0])
        leftmost_label.set_ha('right')  # Align text to the right
        rightmost_label = max(ax.get_xticklabels(), key=lambda x: x.get_position()[0])
        rightmost_label.set_ha('left')  # Align text to the left
        ax.xaxis.set_label_coords(1.2, 0.55)  # Adjust x-label position
    if show_ylabel:
        if latex_var not in ['sin', 'cos', 'theta']:
            var = latex_var.split('_')[0]  # Get the first part of the variable name
            idx = latex_var.split('_')[1]
            ax.set_ylabel(f'$\hat{{{var}}}_{idx}$', fontsize=8, labelpad=2)
        else:
            ax.set_ylabel(f'$\widehat{{{latex_var}}}$', fontsize=8, labelpad=2)
        ax.yaxis.set_tick_params(pad=3)

        # rotate the ylabel to be horizontal
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.set_label_coords(0.5, 1.05)  # Adjust y-label position

    # move the bottom-most yticklabel to the bottom
    ax.yaxis.set_tick_params(pad=1)
    bottommost_label = min(ax.get_yticklabels(), key=lambda x: x.get_position()[1])
    bottommost_label.set_verticalalignment('center')  # Align text to the bottom
    topmost_label = max(ax.get_yticklabels(), key=lambda x: x.get_position()[1])
    topmost_label.set_verticalalignment('center')  # Align text to the top
    
    # Use seaborn to despine (remove top and right spines)
    sns.despine(fig=fig, ax=ax)
    
    # Calculate and add error metrics only if requested
    if show_metrics:
        mse = np.mean((predicted_var - true_var) ** 2)
        r_squared = 1 - np.sum((true_var - predicted_var) ** 2) / np.sum((true_var - np.mean(true_var)) ** 2)
        
        if model_name:
            title = f"{model_name}: {variable_name}\nMSE: {mse:.3f}, R²: {r_squared:.3f}"
        else:
            title = f"{variable_name}\nMSE: {mse:.3f}, R²: {r_squared:.3f}"
            
        ax.set_title(title, fontsize=10, pad=10)
    
    # Save or show figure
    if save_path:
        # Make sure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Save the figure
        fig.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close(fig)
    else:
        plt.show()
    
    return fig, ax
