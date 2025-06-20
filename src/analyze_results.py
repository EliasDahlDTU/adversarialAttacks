import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import glob
import seaborn as sns
from matplotlib.colors import LogNorm
from tqdm import tqdm
from scipy.stats import gaussian_kde
import matplotlib.colors
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple

def load_attack_results(model_name, attack_type):
    """Load attack results for a specific model and attack type from CSV files."""
    data_dir = Path('data/perturbation_analysis')
    
    if attack_type == 'fgsm':
        pattern = f'{model_name}_fgsm_*.csv'
    elif attack_type == 'cw':
        pattern = f'{model_name}_cw_*.csv'
    elif attack_type == 'pgd':
        pattern = f'{model_name}_pgd_*.csv'
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")
    
    attack_files = sorted(glob.glob(str(data_dir / pattern)))
    
    results = {}
    for file_path in tqdm(attack_files, desc=f"Loading {attack_type.upper()} files"):
        # Extract parameter from filename
        param = float(file_path.split('_')[-1].replace('.csv', ''))
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Store results
        results[param] = {
            'true_prob_before': df['true_prob_before'].values,
            'true_prob_after': df['true_prob_after'].values,
            'l2_norm': df['l2_norm'].values,
            'correct_classification': df['correct_classification'].values
        }
    
    return results

def analyze_attack_results(results, attack_type):
    """Analyze the attack results and calculate metrics."""
    all_remaining_prob = []
    all_l2_norms = []
    all_params = []
    robust_accuracies = []
    ra_stds = []
    params = []
    param_name = 'Epsilon' if attack_type == 'fgsm' else 'c'
    
    # Process each parameter threshold
    for param, data in results.items():
        params.append(param)
        
        # Calculate TCCR as 1 - (p_clean-p_adv)/p_clean
        true_prob_drops = data['true_prob_before'] - data['true_prob_after']
        tccr = 1 - (true_prob_drops / data['true_prob_before'])
        
        # Filter out invalid values (where TCCR < 0 or > 1)
        valid_mask = (tccr >= 0) & (tccr <= 1)
        tccr = tccr[valid_mask]
        
        all_remaining_prob.extend(tccr)
        all_l2_norms.extend(data['l2_norm'][valid_mask])
        all_params.extend([param] * len(tccr))
        
        # Calculate Robust Accuracy
        robust_accuracies.append(np.mean(data['correct_classification']))
        ra_stds.append(np.std(data['correct_classification']))
    
    return {
        'params': params,
        'all_remaining_prob': all_remaining_prob,
        'all_l2_norms': all_l2_norms,
        'all_params': all_params,
        'robust_accuracies': robust_accuracies,
        'ra_stds': ra_stds
    }

def plot_results(analysis_results, model_name, attack_type):
    """Plot the Robustness Rate results."""
    params = analysis_results['params']
    param_name = 'Epsilon' if attack_type in ['fgsm', 'pgd'] else 'c'
    
    # Sort params and associated metrics for correct plotting order
    sort_idx = np.argsort(params)
    sorted_params = np.array(params)[sort_idx]
    sorted_ra = np.array(analysis_results['robust_accuracies'])[sort_idx]
    sorted_ra_stds = np.array(analysis_results['ra_stds'])[sort_idx]
    
    # Create figure with three subplots in a horizontal layout
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Set white background for all subplots
    bg_color = 'white'
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(bg_color)
        ax.grid(True, color='gray', alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color('black')
        ax.tick_params(colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
    
    # Plot 1: Combined metrics plot
    # Calculate 95% CI for RA
    ra_n = np.array([np.sum(np.array(analysis_results['all_params']) == p) for p in sorted_params])
    ra_cis = 1.96 * sorted_ra_stds / np.sqrt(ra_n)
    ax1.errorbar(sorted_params, sorted_ra, 
                yerr=ra_cis,
                fmt='o-', color='red', linewidth=2,
                capsize=5, capthick=2, elinewidth=2,
                label='Robust Accuracy')
    
    df = pd.DataFrame({
        param_name: analysis_results['all_params'],
        'TCRR': analysis_results['all_remaining_prob']
    })
    
    # Group by parameter and calculate mean and 95% CI
    grouped = df.groupby(param_name)['TCRR'].agg(['mean', 'std', 'count']).reset_index()
    grouped['ci95'] = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
    
    # Plot mean with 95% CI for TCRR as error bars (pins), not shaded area
    ax1.plot(grouped[param_name], grouped['mean'], 'b-', linewidth=2, label='TCRR')
    # Shaded area for std (RA)
    ra_std_patch = ax1.fill_between(sorted_params, 
                                   sorted_ra - sorted_ra_stds,
                                   sorted_ra + sorted_ra_stds,
                                   alpha=0.15, color='red', label='RA Std')
    # Shaded area for std (TCRR)
    std_patch = ax1.fill_between(grouped[param_name], 
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    alpha=0.15, color='blue', label='TCRR Std')
    # 95% CI error bars
    ax1.errorbar(grouped[param_name], grouped['mean'], yerr=grouped['ci95'], fmt='o', color='blue', capsize=5, capthick=2, elinewidth=2)
    # Add blue dots at the mean TCRR values
    ax1.scatter(grouped[param_name], grouped['mean'], color='blue', s=60, zorder=5)
    
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Value')
    ax1.set_title(f'{model_name} {attack_type.upper()}\nCombined Metrics')
    ax1.set_ylim(0, 1)  # Fix y-axis limits
    ax1.set_xlim(0, max(params))  # Fix x-axis to start at 0

    # Create dummy errorbar handles for legend (dot + vertical error bar + horizontal caps)
    dummy_x = [0]
    dummy_y = [0]
    ra_errbar = ax1.errorbar(dummy_x, dummy_y, yerr=[0.2], fmt='o', color='red', capsize=5, elinewidth=2, capthick=2)
    tcrr_errbar = ax1.errorbar(dummy_x, dummy_y, yerr=[0.2], fmt='o', color='blue', capsize=5, elinewidth=2, capthick=2)

    # Use only the ErrorbarContainer for the legend, and add the std patches
    ax1.legend(
        [ra_errbar, tcrr_errbar, ra_std_patch, std_patch],
        ['Robust Accuracy', 'TCRR', 'RA Std', 'TCRR Std'],
        handler_map={type(ra_errbar): matplotlib.legend_handler.HandlerErrorbar()}
    )

    # Remove the dummy errorbars from the plot
    for err in [ra_errbar, tcrr_errbar]:
        for artist in err.lines:
            if isinstance(artist, (list, tuple)):
                for a in artist:
                    a.set_visible(False)
            else:
                artist.set_visible(False)
    
    # Plot 2: Scatter plot (point cloud)
    if attack_type == 'cw':
        unique_cs = sorted(set(analysis_results['all_params']))
        fgsm_colors = [
            '#4169E1', '#40E0D0', '#50C878', '#228B22', '#32CD32', '#FFFF00', '#FFD700',
            '#FFA07A', '#FF8C00', '#FA8072', '#B22222', '#8B0000', '#DB7093', '#800080'
        ]
        idxs = [0, 1, 3, 5, 7, 9, 11]
        cw_colors = [fgsm_colors[i] for i in idxs[:len(unique_cs)]]
        for c_val, color in zip(unique_cs, cw_colors):
            mask = np.array(analysis_results['all_params']) == c_val
            l2_vals = np.array(analysis_results['all_l2_norms'])[mask]
            tcrr_vals = np.array(analysis_results['all_remaining_prob'])[mask]
            l2_mask = l2_vals <= 9
            ax2.scatter(l2_vals[l2_mask], tcrr_vals[l2_mask],
                       alpha=0.08, s=1, color=color, label=f'c = {c_val:.3g}')
        legend = ax2.legend(bbox_to_anchor=(0.95, 0.95), loc='upper right')
        for handle in legend.legend_handles:
            handle.set_sizes([50])
            handle.set_alpha(1.0)
        plt.setp(legend.get_texts(), color='black')
        ax2.set_xlim(0, 9)  # Cap x-axis at 9 for CW point cloud
    else:
        unique_epsilons = sorted(set(analysis_results['all_params']))
        colors = [
            '#4169E1', '#40E0D0', '#50C878', '#228B22', '#32CD32', '#FFFF00', '#FFD700',
            '#FFA07A', '#FF8C00', '#FA8072', '#B22222', '#8B0000', '#DB7093', '#800080'
        ]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom', colors)
        norm = plt.Normalize(min(unique_epsilons), max(unique_epsilons))
        for eps in unique_epsilons:
            mask = np.array(analysis_results['all_params']) == eps
            color = cmap(norm(eps))
            ax2.scatter(np.array(analysis_results['all_l2_norms'])[mask], 
                       np.array(analysis_results['all_remaining_prob'])[mask],
                       alpha=0.08, s=1, color=color, label=f'ε = {eps:.2f}')
        legend = ax2.legend(bbox_to_anchor=(0.95, 0.95), loc='upper right')
        for handle in legend.legend_handles:
            handle.set_sizes([50])
            handle.set_alpha(1.0)
        plt.setp(legend.get_texts(), color='black')
    
    ax2.set_xlabel('L2 Perturbation Size')
    ax2.set_ylabel('TCRR')
    ax2.set_title(f'{model_name} {attack_type.upper()}\nPoint Cloud')
    ax2.set_ylim(0, 1)  # Fix y-axis limits
    if attack_type == 'cw':
        ax2.set_xlim(0, 9)
    else:
        ax2.set_xlim(0, max(analysis_results['all_l2_norms']))
    
    # Plot 3: 2D Histogram
    if attack_type == 'cw':
        # Cap L2 values at 9 for fair comparison
        l2_vals = np.array(analysis_results['all_l2_norms'])
        tcrr_vals = np.array(analysis_results['all_remaining_prob'])
        l2_mask = l2_vals <= 9
        h3 = ax3.hist2d(
            l2_vals[l2_mask],
            tcrr_vals[l2_mask],
            bins=100,
            cmap='plasma',
            norm=LogNorm(vmin=1, vmax=None)
        )
        ax3.set_xlim(0, 9)
    else:
        h3 = ax3.hist2d(
            analysis_results['all_l2_norms'],
            analysis_results['all_remaining_prob'],
            bins=100,
            cmap='plasma',
            norm=LogNorm(vmin=1, vmax=None)
        )
        ax3.set_xlim(0, max(analysis_results['all_l2_norms']))
    ax3.set_xlabel('L2 Perturbation Size')
    ax3.set_ylabel('TCRR')
    ax3.set_title(f'{model_name} {attack_type.upper()}\nDensity Plot')
    ax3.set_ylim(0, 1)  # Fix y-axis limits
    cbar = plt.colorbar(h3[3], ax=ax3, label='Count (log scale)')
    cbar.ax.yaxis.label.set_color('black')
    cbar.ax.tick_params(colors='black')
    
    # Set figure background to white
    fig.patch.set_facecolor(bg_color)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_{attack_type}_tcrr.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def main():
    # Create results directory if it doesn't exist
    Path('results').mkdir(exist_ok=True)
    
    # Analyze both models and attack types
    for model_name in ['vgg16', 'resnet50']:
        for attack_type in ['fgsm', 'pgd', 'cw']:
            print(f"\nAnalyzing {model_name} with {attack_type.upper()}...")
            
            # Load results
            results = load_attack_results(model_name, attack_type)
            
            if not results:
                print(f"No results found for {model_name} {attack_type.upper()}.")
                continue
            
            # Analyze results
            analysis_results = analyze_attack_results(results, attack_type)
            
            # Plot results
            plot_results(analysis_results, model_name, attack_type)
            
            # Print summary statistics
            param_name = 'Epsilon' if attack_type in ['fgsm', 'pgd'] else 'c'  # Define param_name here
            print(f"\n{model_name.upper()} {attack_type.upper()} Summary:")
            print("=" * 40)
            print("\nRobust Accuracy Summary:")
            for i, param in enumerate(analysis_results['params']):
                print(f"{param_name}: {param:.3f} - RA: {analysis_results['robust_accuracies'][i]:.4f} ± {analysis_results['ra_stds'][i]:.4f}")
            
            print("\nTCRR Summary:")
            print(f"Mean: {np.mean(analysis_results['all_remaining_prob']):.4f}")
            print(f"Median: {np.median(analysis_results['all_remaining_prob']):.4f}")
            print(f"Std: {np.std(analysis_results['all_remaining_prob']):.4f}")
            print(f"Min: {np.min(analysis_results['all_remaining_prob']):.4f}")
            print(f"Max: {np.max(analysis_results['all_remaining_prob']):.4f}")

if __name__ == '__main__':
    main() 