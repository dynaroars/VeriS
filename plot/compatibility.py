import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from .utils import *

def main():
    dfs = []
    # join two dfs
    verifiers = {
        'neuralsat': r'\textsc{NeuralSat}',
        'abcrown': r'\textsc{$\alpha\beta$-Crown (I)}',
        'abcrown_A': r'\textsc{$\alpha\beta$-Crown (N)}',
    }
    for verifier in verifiers.keys():
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), f'{verifier}_results.csv'))
        df['verifier'] = verifier
        dfs.append(df)    
    dfs = pd.concat(dfs)
    
    # for benchmark_type in df['benchmark_type'].unique():
    # bar plot for each perturbation type compared among verifiers
    fig_sizes = {
        'time_invariant': (5, 3),
        'time_varying': (3, 3),
    }
    for benchmark_type in df['benchmark_type'].unique():
        df = dfs[dfs['benchmark_type'] == benchmark_type]
        plot_data = []
        for perturbation_type in df['perturbation_type'].unique():
            for verifier in verifiers.keys():
                tmp_df = df[df['perturbation_type'] == perturbation_type]
                tmp_df = tmp_df[tmp_df['verifier'] == verifier]
                unsat_tmp_df = tmp_df[tmp_df['status'] == 'unsat']
                sat_tmp_df = tmp_df[tmp_df['status'] == 'sat']
                total = tmp_df['status'].value_counts().sum()
                solved = sat_tmp_df['status'].value_counts().sum() + unsat_tmp_df['status'].value_counts().sum()
                print(f'{perturbation_type=} {verifier=} {solved=}')
                plot_data.append({
                    'verifier': verifiers[verifier],
                    'perturbation': perturbation_type.capitalize().replace('Motion_blur', 'Blur'),
                    'solved': solved / total * 100
                })
                # bar plot side by side for each verifier
                
        df_plot = pd.DataFrame(plot_data)
        print(df_plot)
        
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.0)

        # Create the plot
        fig, ax = plt.subplots(figsize=fig_sizes[benchmark_type])

        # Create the grouped bar plot
        bar_plot = sns.barplot(
            data=df_plot, 
            x='perturbation', 
            y='solved', 
            hue='verifier', 
            palette='muted',
            ax=ax,
            dodge=True,
        )

        # Add value labels on bars
        for container in bar_plot.containers:
            bar_plot.bar_label(container, fmt='%d')
            
                # Customize the plot
        ax.set_ylim(0, 110)
        ax.set_xlabel('')
        ax.set_ylabel('Solved Instances (\%)')
        if benchmark_type == 'time_varying':
            ax.legend(
                loc="upper right",
                fancybox=True,
                # ncol=len(verifiers),
                # bbox_to_anchor=(0.5, 1.15),
            )
        else:
            ax.legend_ = None

        # Make layout tight
        # plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'../../figure/compatibility_{benchmark_type}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
if __name__ == "__main__":
    main()