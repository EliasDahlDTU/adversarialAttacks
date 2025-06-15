import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_robustness_vs_tolerance(csv_path="results/robustness_results.csv",
                                 out_dir="plots"):
    # 1) Load and filter
    df = pd.read_csv(csv_path)
    df = df[df['metric'].isin(['RA_tol', 'RR_tol'])]

    # 2) Make sure output dir exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) Plot per attack
    for atk in df['attack'].unique():
        sub = df[df['attack'] == atk].sort_values('tolerance')

        fig, ax = plt.subplots(figsize=(6,4))
        for metric in ['RA_tol', 'RR_tol']:
            part = sub[sub['metric'] == metric]
            ax.plot(part['tolerance'], part['value'],
                    marker='o', label=metric)

        ax.set_xlabel("Perturbation tolerance ε")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, 1)
        ax.set_title(f"{atk}: RA and RR vs ε")
        ax.grid(True)
        ax.legend()

        save_path = out_dir / f"{atk}_vs_tolerance.png"
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_robustness_vs_tolerance()
