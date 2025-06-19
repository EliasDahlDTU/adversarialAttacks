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
        
        # Debug prints for first parameter
        if param == min(params):
            print(f"\nDebug for {param_name} = {param}:")
            print(f"Mean TCCR: {np.mean(tccr):.4f}")
            print(f"Min TCCR: {np.min(tccr):.4f}")
            print(f"Max TCCR: {np.max(tccr):.4f}")
            print(f"Number of valid samples: {np.sum(valid_mask)}")
        
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
    ax1.errorbar(sorted_params, sorted_ra, 
                yerr=sorted_ra_stds,
                fmt='o-', color='red', linewidth=2,
                capsize=5, capthick=2, elinewidth=2,
                label='Robust Accuracy')
    
    df = pd.DataFrame({
        param_name: analysis_results['all_params'],
        'TCCR': analysis_results['all_remaining_prob']
    })
    
    # Group by parameter and calculate mean and std
    grouped = df.groupby(param_name)['TCCR'].agg(['mean', 'std']).reset_index()
    
    # Plot mean with std
    ax1.plot(grouped[param_name], grouped['mean'], 'b-', linewidth=2, label='TCCR')
    ax1.fill_between(grouped[param_name], 
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    alpha=0.2, color='blue')
    
    # Add blue dots at the mean TCCR values
    ax1.scatter(grouped[param_name], grouped['mean'], color='blue', s=60, zorder=5)
    
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Value')
    ax1.set_title(f'{model_name} {attack_type.upper()}\nCombined Metrics')
    ax1.set_ylim(0, 1)  # Fix y-axis limits
    ax1.set_xlim(0, max(params))  # Fix x-axis to start at 0
    legend1 = ax1.legend()
    plt.setp(legend1.get_texts(), color='black')
    
    # Plot 2: Scatter plot (point cloud)
    if attack_type == 'cw':
        unique_cs = sorted(set(analysis_results['all_params']))
        # Use the 1st, 2nd, 4th, 6th, 8th, 10th, and 12th colors from the FGSM color list
        fgsm_colors = [
            '#4169E1',  # 1 Royal Blue
            '#40E0D0',  # 2 Turquoise
            '#50C878',  # 3 Emerald Green
            '#228B22',  # 4 Forest Green
            '#32CD32',  # 5 Lime Green
            '#FFFF00',  # 6 Cool Yellow
            '#FFD700',  # 7 Regular Yellow
            '#FFA07A',  # 8 Light Orange
            '#FF8C00',  # 9 Orange
            '#FA8072',  # 10 Salmon
            '#B22222',  # 11 Brick Colored
            '#8B0000',  # 12 Deep Blood Red
            '#DB7093',  # 13 Pinkish Violet
            '#800080'   # 14 Purple
        ]
        idxs = [0, 1, 3, 5, 7, 9, 11]  # 1,2,4,6,8,10,12 (0-based)
        cw_colors = [fgsm_colors[i] for i in idxs[:len(unique_cs)]]
        for c_val, color in zip(unique_cs, cw_colors):
            mask = np.array(analysis_results['all_params']) == c_val
            ax2.scatter(np.array(analysis_results['all_l2_norms'])[mask], 
                       np.array(analysis_results['all_remaining_prob'])[mask],
                       alpha=0.08, s=1, color=color, label=f'c = {c_val:.3g}')
        legend = ax2.legend(bbox_to_anchor=(0.95, 0.95), loc='upper right')
        for handle in legend.legend_handles:
            handle.set_sizes([50])
            handle.set_alpha(1.0)
        plt.setp(legend.get_texts(), color='black')
    else:
        # For FGSM, use a custom color progression with balanced transitions
        unique_epsilons = sorted(set(analysis_results['all_params']))
        # Create a custom colormap with balanced color distribution
        colors = [
            '#4169E1',  # Royal Blue (brighter than deep cold blue)
            '#40E0D0',  # Turquoise
            '#50C878',  # Emerald Green
            '#228B22',  # Forest Green
            '#32CD32',  # Lime Green
            '#FFFF00',  # Cool Yellow
            '#FFD700',  # Regular Yellow
            '#FFA07A',  # Light Orange
            '#FF8C00',  # Orange
            '#FA8072',  # Salmon
            '#B22222',  # Brick Colored
            '#8B0000',  # Deep Blood Red
            '#DB7093',  # Pinkish Violet
            '#800080'   # Purple
        ]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom', colors)
        # Normalize epsilon values to [0,1] for colormap
        norm = plt.Normalize(min(unique_epsilons), max(unique_epsilons))
        
        # Plot each epsilon value separately with its color from the colormap
        for eps in unique_epsilons:
            mask = np.array(analysis_results['all_params']) == eps
            color = cmap(norm(eps))
            ax2.scatter(np.array(analysis_results['all_l2_norms'])[mask], 
                       np.array(analysis_results['all_remaining_prob'])[mask],
                       alpha=0.08, s=1, color=color, label=f'ε = {eps:.2f}')
        
        # Add legend inside the plot with larger markers
        legend = ax2.legend(bbox_to_anchor=(0.95, 0.95), loc='upper right')
        # Make the legend markers larger and solid
        for handle in legend.legend_handles:
            handle.set_sizes([50])  # Increase marker size in legend
            handle.set_alpha(1.0)   # Make legend markers solid
        plt.setp(legend.get_texts(), color='black')
    
    ax2.set_xlabel('L2 Perturbation Size')
    ax2.set_ylabel('TCCR')
    ax2.set_title(f'{model_name} {attack_type.upper()}\nPoint Cloud')
    ax2.set_ylim(0, 1)  # Fix y-axis limits
    ax2.set_xlim(0, max(analysis_results['all_l2_norms']))  # Fix x-axis to start at 0
    
    # Plot 3: 2D Histogram
    h3 = ax3.hist2d(
        analysis_results['all_l2_norms'],
        analysis_results['all_remaining_prob'],
        bins=100,
        cmap='plasma',
        norm=LogNorm(vmin=1, vmax=None)
    )
    ax3.set_xlabel('L2 Perturbation Size')
    ax3.set_ylabel('TCCR')
    ax3.set_title(f'{model_name} {attack_type.upper()}\nDensity Plot')
    ax3.set_ylim(0, 1)  # Fix y-axis limits
    ax3.set_xlim(0, max(analysis_results['all_l2_norms']))  # Fix x-axis to start at 0
    cbar = plt.colorbar(h3[3], ax=ax3, label='Count (log scale)')
    cbar.ax.yaxis.label.set_color('black')
    cbar.ax.tick_params(colors='black')
    
    # Set figure background to white
    fig.patch.set_facecolor(bg_color)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_{attack_type}_tccr.png', bbox_inches='tight', facecolor='white')
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
            
            print("\nTCCR Summary:")
            print(f"Mean: {np.mean(analysis_results['all_remaining_prob']):.4f}")
            print(f"Median: {np.median(analysis_results['all_remaining_prob']):.4f}")
            print(f"Std: {np.std(analysis_results['all_remaining_prob']):.4f}")
            print(f"Min: {np.min(analysis_results['all_remaining_prob']):.4f}")
            print(f"Max: {np.max(analysis_results['all_remaining_prob']):.4f}")

if __name__ == '__main__':
    main() 