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
    for benchmark_type in df['benchmark_type'].unique():
        print()
        print('='*80)
        print('Generating table for', benchmark_type)
        print('='*80)
        print()
        perturbation_types = ['motion_blur_0', 'motion_blur_45', 'motion_blur_90']
        task = 'image'
        print(f'\multirow{{4}}{{*}}{{{task.capitalize()}}}')
        strengths = df['strength'][df['task'] == task][df['benchmark_type'] == benchmark_type].unique()
        for strength in sorted(strengths):
            line = f"& $[0.0, {strength}]$ & "
            for perturbation_type in perturbation_types:
                row = df[(df['strength'] == strength) & (df['perturbation_type'] == perturbation_type) & (df['task'] == task)]
                unsat = len(row[row['status'] == 'unsat'])
                sat = len(row[row['status'] == 'sat'])
                timeout = len(row) - (unsat + sat)
                runtime = row['runtime'].sum()
                line += f" {unsat+sat}/{timeout} & "
            line = line[:-2] + '\\\\'
            print(line)
                
            if strength != strengths[-1]:
                print('\\cmidrule(l){2-5}')
        # print total row
        line = r"\textbf{Total} & & "
        for perturbation_type in perturbation_types:
            row = df[(df['perturbation_type'] == perturbation_type)]
            unsat = len(row[row['status'] == 'unsat'])
            sat = len(row[row['status'] == 'sat'])
            timeout = len(row) - (unsat + sat)
            runtime = row['runtime'].sum()
            line += f"{unsat+sat}/{timeout} & "
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
            print(f'\multirow{{4}}{{*}}{{{task.upper()}}}')
            strengths = sorted(df['strength'][df['task'] == task][df['benchmark_type'] == benchmark_type].unique())
            for strength in strengths:
                line = f"& $[0.0, {strength}]$ & "
                for perturbation_type in perturbation_types:
                    # print(strength, perturbation_type)
                    row = df[(df['strength'] == strength) & (df['perturbation_type'] == perturbation_type) & (df['task'] == task)]
                    # print(benchmark_type, perturbation_type, len(row), row['task'].unique())
                    unsat = len(row[row['status'] == 'unsat'])
                    sat = len(row[row['status'] == 'sat'])
                    timeout = len(row) - (unsat + sat)
                    runtime = row['runtime'].sum()
                    line += f" {unsat+sat}/{timeout} & "
                # print(row)
                line = line[:-2] + '\\\\'
                print(line)
                # exit()
                # print(row)
                if strength != strengths[-1]:
                    print('\\cmidrule(l){2-5}')
            if task != df['task'].unique()[-1]:
                print('\\midrule')
        # print total row
        # \textbf{Total} & & & 484/121/43 & 3969.2 &  485/96/67 & 4815.9  & 513/87/48 & 4212.9 \\
        line = r"\textbf{Total} & & "
        for perturbation_type in perturbation_types:
            row = df[(df['perturbation_type'] == perturbation_type)]
            unsat = len(row[row['status'] == 'unsat'])
            sat = len(row[row['status'] == 'sat'])
            timeout = len(row) - (unsat + sat)
            runtime = row['runtime'].sum()
            
            line += f"{unsat+sat}/{timeout} & "
        line = line[:-2] + '\\\\'
        print(line)
        print('\\bottomrule')
    
def print_table_baseline(args):
    df_maximal = pd.read_csv(args.csv)
    df_maximal['verifier'] = 'maximal'
    df_sr = pd.read_csv(args.csv.replace('_baseline', ''))
    df_sr['verifier'] = 'sr'
    df = pd.concat([df_maximal, df_sr])
    # convert strength to float
    
    verifiers = ['Maximal', 'SR']
    for benchmark_type in ['time_invariant']:
        # print(benchmark_type)
        perturbation_types = df_sr[df_sr['benchmark_type'] == benchmark_type]['perturbation_type'].unique()
        print(perturbation_types)
        for verifier in verifiers:
            # print(f'\multirow{{4}}{{*}}{{{verifier}}}')
            # strengths = sorted(df_sr[df_sr['benchmark_type'] == benchmark_type]['strength'].unique())
            # strengths = [0.1, 0.5, 1.0]
            line = f"{verifier} & "
            # print(strengths)
            # for strength in strengths:
            for perturbation_type in perturbation_types:
                df_tmp = df[(df['verifier'] == verifier.lower()) & (df['benchmark_type'] == benchmark_type) & (df['perturbation_type'] == perturbation_type)]
                # print(df_tmp)
                # exit()
                unsat = len(df_tmp[df_tmp['status'] == 'unsat'])
                sat = len(df_tmp[df_tmp['status'] == 'sat'])
                timeout = len(df_tmp) - (unsat + sat)
                    # print(f'{perturbation_type=} {verifier=} {unsat}/{sat}/{timeout} {runtime:.1f}')
                line += f" {unsat}/{sat}/{timeout} & "
            line = line[:-2] + '\\\\'
            print(line)
            if verifier != verifiers[-1]:
                print('\\midrule')
            else:
                print('\\bottomrule')
    
    
            
def print_table_unoptimized(args):
    df_unoptimized = pd.read_csv(args.csv)
    df_unoptimized['verifier'] = 'unoptimized'  
    df_sr = pd.read_csv(args.csv.replace('_unoptimized', ''))
    df_sr['verifier'] = 'optimized'
    df = pd.concat([df_unoptimized, df_sr])
    
    verifiers = ['Unoptimized', 'Optimized']
    
    for benchmark_type in df_unoptimized['benchmark_type'].unique():
        print(benchmark_type)
        for verifier in verifiers:
            # print(f'\multirow{{4}}{{*}}{{{verifier}}}')
            strengths = sorted(df_unoptimized['strength'].unique())
            line = f"{verifier} & "
            
            for strength in strengths:
                df_tmp = df[(df['verifier'] == verifier.lower()) & (df['strength'] == strength) & (df['benchmark_type'] == benchmark_type)]
                unsat = len(df_tmp[df_tmp['status'] == 'unsat'])
                sat = len(df_tmp[df_tmp['status'] == 'sat'])
                timeout = len(df_tmp) - (unsat + sat)
                    # print(f'{perturbation_type=} {verifier=} {unsat}/{sat}/{timeout} {runtime:.1f}')
                line += f" {unsat+sat}/{timeout} & "
            line = line[:-2] + '\\\\'
            print(line)
            if verifier != verifiers[-1]:
                print('\\midrule')
            else:
                print('\\bottomrule')
    
    
    
def print_table_aggregated(args):
    df = pd.read_csv(args.csv)
    for benchmark_type in df['benchmark_type'].unique():
        print(benchmark_type)
        tasks = df['task'].unique()
        for task in tasks:
            # print(f'\multirow{{4}}{{*}}{{{verifier}}}')
            strengths = [0.1, 0.5, 1.0] if benchmark_type == 'time_invariant' else [0.1, 0.2, 0.3]
            line = f"{task} & "
            
            for strength in strengths:
                df_tmp = df[(df['task'] == task) & (df['strength'] == strength) & (df['benchmark_type'] == benchmark_type)]
                unsat = len(df_tmp[df_tmp['status'] == 'unsat'])
                sat = len(df_tmp[df_tmp['status'] == 'sat'])
                timeout = len(df_tmp) - (unsat + sat)
                    # print(f'{perturbation_type=} {verifier=} {unsat}/{sat}/{timeout} {runtime:.1f}')
                line += f" {unsat}/{sat}/{timeout} & "
            line = line[:-2] + '\\\\'
            print(line)
            if task != tasks[-1]:
                print('\\midrule')
            else:
                print('\\bottomrule')
if __name__ == "__main__":
    args = parse_args()
    # print_table_image(args)
    # print_table_kws_ecg(args)
    # print_table_baseline(args)
    # print_table_unoptimized(args)
    print_table_aggregated(args)