import pandas as pd
import argparse
import os


# \begin{table}[]
# \begin{tabular}{@{}ccccclll@{}}
# \toprule
#             &          & Lowpass &      & Echo  &      & Sharpen &      \\ \cmidrule(l){3-8} 
# Kernel Size & Strength & U/S/T   & Time & U/S/T & Time & U/S/T   & Time \\ \midrule
#             &          &         &      &       &      &         &      \\
#             &          &         &      &       &      &         &      \\
#             &          &         &      &       &      &         &      \\ \bottomrule
# \end{tabular}
# \caption{}
# \label{tab:my-table}
# \end{table}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="CSV file")
    args = p.parse_args()
    return args

def print_table_image(args):
    df = pd.read_csv(args.csv)
    print(len(df))
    for benchmark_type in df['benchmark_type'].unique():
        print()
        print('='*80)
        print('Generating table for', benchmark_type)
        print('='*80)
        print()
        perturbation_types = ['motion_blur_0', 'motion_blur_45', 'motion_blur_90']
        task = 'image'
        print(f'\multirow{{9}}{{*}}{{{task.upper()}}}')
        kernel_sizes = df['kernel_size'][df['task'] == task][df['benchmark_type'] == benchmark_type].unique()
        for kernel_size in sorted(kernel_sizes):
            strengths = df['strength'][df['task'] == task][df['benchmark_type'] == benchmark_type][df['kernel_size'] == kernel_size].unique()
            for strength in sorted(strengths):
                line = f"& {kernel_size} & {strength} & "
                for perturbation_type in perturbation_types:
                    row = df[(df['strength'] == strength) & (df['perturbation_type'] == perturbation_type) & (df['task'] == task) & (df['kernel_size'] == kernel_size)]
                    unsat = len(row[row['status'] == 'unsat'])
                    sat = len(row[row['status'] == 'sat'])
                    timeout = len(row) - (unsat + sat)
                    runtime = row['runtime'].sum()
                    line += f" {unsat}/{sat}/{timeout} & {runtime:.1f} &"
                line = line[:-1] + '\\\\'
                print(line)
                
            if kernel_size != kernel_sizes[-1]:
                print('\\cmidrule(l){2-9}')
        # print total row
        line = r"\textbf{Total} & & & "
        for perturbation_type in perturbation_types:
            row = df[(df['perturbation_type'] == perturbation_type)]
            unsat = len(row[row['status'] == 'unsat'])
            sat = len(row[row['status'] == 'sat'])
            timeout = len(row) - (unsat + sat)
            runtime = row['runtime'].sum()
            line += f"{unsat}/{sat}/{timeout} & {runtime:.1f} & "
        line = line[:-2] + '\\\\'
        print('\\midrule')
        print(line)
        print('\\bottomrule')

def print_table_kws_ecg(args):
    df = pd.read_csv(args.csv)
    print(len(df))
    for benchmark_type in df['benchmark_type'].unique():
        print()
        print('='*80)
        print('Generating table for', benchmark_type)
        print('='*80)
        print()
        perturbation_types = ['lowpass', 'echo', 'highpass'] if benchmark_type == 'time_invariant' else ['linear', 'sinusoidal', 'gaussian']
        for task in ['ecg', 'kws']:
            print(f'\multirow{{9}}{{*}}{{{task.upper()}}}')
            kernel_sizes = df['kernel_size'][df['task'] == task][df['benchmark_type'] == benchmark_type].unique()
            for kernel_size in sorted(kernel_sizes):
                strengths = df['strength'][df['task'] == task][df['benchmark_type'] == benchmark_type][df['kernel_size'] == kernel_size].unique()
                for strength in sorted(strengths):
                    line = f"& {kernel_size} & {strength} & "
                    for perturbation_type in perturbation_types:
                        # print(strength, perturbation_type)
                        row = df[(df['strength'] == strength) & (df['perturbation_type'] == perturbation_type) & (df['task'] == task) & (df['kernel_size'] == kernel_size)]
                        # print(benchmark_type, perturbation_type, len(row), row['task'].unique())
                        unsat = len(row[row['status'] == 'unsat'])
                        sat = len(row[row['status'] == 'sat'])
                        timeout = len(row) - (unsat + sat)
                        runtime = row['runtime'].sum()
                        line += f" {unsat}/{sat}/{timeout} & {runtime:.1f} &"
                    # print(row)
                    line = line[:-1] + '\\\\'
                    print(line)
                    # exit()
                    # print(row)
                if kernel_size == kernel_sizes[-1]:
                    if task == df['task'].unique()[-1]:
                        # print('\\bottomrule')
                        pass
                    else:
                        print('\\midrule')
                else:   
                    print('\\cmidrule(l){2-9}')
    
        # print total row
        # \textbf{Total} & & & 484/121/43 & 3969.2 &  485/96/67 & 4815.9  & 513/87/48 & 4212.9 \\
        line = r"\textbf{Total} & & & "
        for perturbation_type in perturbation_types:
            row = df[(df['perturbation_type'] == perturbation_type)]
            unsat = len(row[row['status'] == 'unsat'])
            sat = len(row[row['status'] == 'sat'])
            timeout = len(row) - (unsat + sat)
            runtime = row['runtime'].sum()
            
            line += f"{unsat}/{sat}/{timeout} & {runtime:.1f} & "
        line = line[:-2] + '\\\\'
        print('\\midrule')
        print(line)
        print('\\bottomrule')
    
if __name__ == "__main__":
    args = parse_args()
    # print_table_kws_ecg(args)
    print_table_image(args)