import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import glob
# import seaborn as sns
from matplotlib.colors import LogNorm
from tqdm import tqdm
# from scipy.stats import gaussian_kde
import matplotlib.colors
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple

def load_transferability_results(src_model_name, tgt_model_name, attack_type):
    """Load transferability results for a specific source-target model pair and attack type from CSV files."""
    data_dir = Path('data/transferability')
    
    # Pattern for transferability files: {SrcModel}_trans_to_{TgtModel}_{attack_type}_*.csv
    pattern = f'{src_model_name.capitalize()}_trans_to_{tgt_model_name.capitalize()}_{attack_type}_*.csv'
    
    attack_files = sorted(glob.glob(str(data_dir / pattern)))
    
    results = {}
    for file_path in tqdm(attack_files, desc=f"Loading {attack_type.upper()} files"):
        # Extract parameter from filename
        param = float(file_path.split('_')[-1].replace('.csv', ''))
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Store results
        results[param] = {
            'adv_correct_classification_tgt': df['adv_correct_classification_tgt'].values,
            'clean_correct_classification': df['clean_correct_classification'].values
        }
    
    return results

def analyze_transferability_results(results, attack_type):
    """Analyze the transferability results and calculate metrics."""
    all_params = []
    robust_accuracies = []
    ra_stds = []
    ra_sample_sizes = []  # Track actual sample sizes used for RA calculation
    params = []
    param_name = 'Epsilon' if attack_type in ['fgsm', 'pgd'] else 'c'
    
    # Process each parameter threshold
    for param, data in results.items():
        params.append(param)
        all_params.extend([param] * len(data['adv_correct_classification_tgt']))
        
        # Calculate Robust Accuracy - only for images that were correctly classified when clean
        clean_correct_mask = data['clean_correct_classification'] == True
        ra_sample_size = np.sum(clean_correct_mask)  # Track actual sample size
        ra_sample_sizes.append(ra_sample_size)
        
        if ra_sample_size > 0:
            ra = np.mean(data['adv_correct_classification_tgt'][clean_correct_mask])
        else:
            ra = 0.0  # If no clean images were correct, RA is 0
        robust_accuracies.append(ra)
        
        # Calculate RA std - only for images that were correctly classified when clean
        if ra_sample_size > 0:
            ra_std = np.std(data['adv_correct_classification_tgt'][clean_correct_mask])
        else:
            ra_std = 0.0
        ra_stds.append(ra_std)
    
    return {
        'params': params,
        'all_params': all_params,
        'robust_accuracies': robust_accuracies,
        'ra_stds': ra_stds,
        'ra_sample_sizes': ra_sample_sizes  # Add sample sizes to return dict
    }

def plot_combined_transferability_results(all_results, src_model_name, tgt_model_name):
    """Plot combined transferability results for all attack types."""
    # Create figure with three subplots
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
    
    # Colors for different attack types
    attack_colors = {
        'fgsm': '#4169E1',  # Blue
        'pgd': '#FF6B6B',   # Red
        'cw': '#32CD32'     # Green
    }
    
    attack_markers = {
        'fgsm': 'o',
        'pgd': 's',
        'cw': '^'
    }
    
    # Plot 1: FGSM
    if 'fgsm' in all_results:
        results = all_results['fgsm']
        # Sort params and associated metrics for correct plotting order
        sort_idx = np.argsort(results['params'])
        sorted_params = np.array(results['params'])[sort_idx]
        sorted_ra = np.array(results['robust_accuracies'])[sort_idx]
        sorted_ra_stds = np.array(results['ra_stds'])[sort_idx]
        sorted_ra_sample_sizes = np.array(results['ra_sample_sizes'])[sort_idx]
        
        # Calculate 95% CI for RA using actual sample sizes used for RA calculation
        ra_cis = 1.96 * sorted_ra_stds / np.sqrt(sorted_ra_sample_sizes)
        # Handle division by zero (when sample size is 0)
        ra_cis = np.where(sorted_ra_sample_sizes > 0, ra_cis, 0.0)
        
        # Plot error bars (confidence intervals)
        ax1.errorbar(sorted_params, sorted_ra, 
                    yerr=ra_cis,
                    fmt=f'{attack_markers["fgsm"]}-', 
                    color=attack_colors['fgsm'], 
                    linewidth=2, markersize=6,
                    capsize=5, capthick=2, elinewidth=2,
                    label='FGSM')
        
        # Plot standard deviation bands
        ax1.fill_between(sorted_params, 
                        sorted_ra - sorted_ra_stds,
                        sorted_ra + sorted_ra_stds,
                        alpha=0.15, color=attack_colors['fgsm'], label='FGSM Std')
    
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Robust Accuracy')
    ax1.set_title(f'{src_model_name.upper()} → {tgt_model_name.upper()}\nFGSM Transferability')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower left')
    
    # Plot 2: PGD
    if 'pgd' in all_results:
        results = all_results['pgd']
        # Sort params and associated metrics for correct plotting order
        sort_idx = np.argsort(results['params'])
        sorted_params = np.array(results['params'])[sort_idx]
        sorted_ra = np.array(results['robust_accuracies'])[sort_idx]
        sorted_ra_stds = np.array(results['ra_stds'])[sort_idx]
        sorted_ra_sample_sizes = np.array(results['ra_sample_sizes'])[sort_idx]
        
        # Calculate 95% CI for RA using actual sample sizes used for RA calculation
        ra_cis = 1.96 * sorted_ra_stds / np.sqrt(sorted_ra_sample_sizes)
        # Handle division by zero (when sample size is 0)
        ra_cis = np.where(sorted_ra_sample_sizes > 0, ra_cis, 0.0)
        
        # Plot error bars (confidence intervals)
        ax2.errorbar(sorted_params, sorted_ra, 
                    yerr=ra_cis,
                    fmt=f'{attack_markers["pgd"]}-', 
                    color=attack_colors['pgd'], 
                    linewidth=2, markersize=6,
                    capsize=5, capthick=2, elinewidth=2,
                    label='PGD')
        
        # Plot standard deviation bands
        ax2.fill_between(sorted_params, 
                        sorted_ra - sorted_ra_stds,
                        sorted_ra + sorted_ra_stds,
                        alpha=0.15, color=attack_colors['pgd'], label='PGD Std')
    
    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Robust Accuracy')
    ax2.set_title(f'{src_model_name.upper()} → {tgt_model_name.upper()}\nPGD Transferability')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='lower left')
    
    # Plot 3: CW
    if 'cw' in all_results:
        results = all_results['cw']
        # Sort params and associated metrics for correct plotting order
        sort_idx = np.argsort(results['params'])
        sorted_params = np.array(results['params'])[sort_idx]
        sorted_ra = np.array(results['robust_accuracies'])[sort_idx]
        sorted_ra_stds = np.array(results['ra_stds'])[sort_idx]
        sorted_ra_sample_sizes = np.array(results['ra_sample_sizes'])[sort_idx]
        
        # Calculate 95% CI for RA using actual sample sizes used for RA calculation
        ra_cis = 1.96 * sorted_ra_stds / np.sqrt(sorted_ra_sample_sizes)
        # Handle division by zero (when sample size is 0)
        ra_cis = np.where(sorted_ra_sample_sizes > 0, ra_cis, 0.0)
        
        # Plot error bars (confidence intervals)
        ax3.errorbar(sorted_params, sorted_ra, 
                    yerr=ra_cis,
                    fmt=f'{attack_markers["cw"]}-', 
                    color=attack_colors['cw'], 
                    linewidth=2, markersize=6,
                    capsize=5, capthick=2, elinewidth=2,
                    label='CW')
        
        # Plot standard deviation bands
        ax3.fill_between(sorted_params, 
                        sorted_ra - sorted_ra_stds,
                        sorted_ra + sorted_ra_stds,
                        alpha=0.15, color=attack_colors['cw'], label='CW Std')
    
    ax3.set_xlabel('c')
    ax3.set_ylabel('Robust Accuracy')
    ax3.set_title(f'{src_model_name.upper()} → {tgt_model_name.upper()}\nCW Transferability')
    ax3.set_ylim(0, 1)
    ax3.set_xscale('log')
    ax3.legend(loc='lower left')
    
    # Set figure background to white
    fig.patch.set_facecolor(bg_color)
    
    plt.tight_layout()
    plt.savefig(f'results/{src_model_name}_to_{tgt_model_name}_transferability.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def main():
    # Create results directory if it doesn't exist
    Path('results').mkdir(exist_ok=True)
    
    # Analyze both transfer directions
    transfer_directions = [
        ('vgg16', 'resnet50'),
        ('resnet50', 'vgg16')
    ]
    
    for src_model, tgt_model in transfer_directions:
        print(f"\nAnalyzing {src_model.upper()} → {tgt_model.upper()} transferability...")
        
        all_results = {}
        
        # Load results for all attack types
        for attack_type in ['fgsm', 'pgd', 'cw']:
            print(f"  Loading {attack_type.upper()} results...")
            
            # Load results
            results = load_transferability_results(src_model, tgt_model, attack_type)
            
            if not results:
                print(f"    No results found for {attack_type.upper()}.")
                continue
            
            # Analyze results
            analysis_results = analyze_transferability_results(results, attack_type)
            all_results[attack_type] = analysis_results
        
        if not all_results:
            print(f"No results found for {src_model.upper()} → {tgt_model.upper()}.")
            continue
        
        # Plot combined results
        plot_combined_transferability_results(all_results, src_model, tgt_model)
        
        # Print summary statistics
        print(f"\n{src_model.upper()} → {tgt_model.upper()} Summary:")
        print("=" * 50)
        for attack_type, results in all_results.items():
            param_name = 'Epsilon' if attack_type in ['fgsm', 'pgd'] else 'c'
            print(f"\n{attack_type.upper()} Summary:")
            for i, param in enumerate(results['params']):
                print(f"  {param_name}: {param:.3f} - RA: {results['robust_accuracies'][i]:.4f} ± {results['ra_stds'][i]:.4f}")

if __name__ == '__main__':
    main() 