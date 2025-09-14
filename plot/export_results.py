import pandas as pd
import argparse
import os



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_dir", type=str, default="generated_benchmark/", help="Root directory for benchmark")
    p.add_argument("--output_dir", type=str, default='../exp/results/', help="Output directory")
    p.add_argument("--postfix", type=str, default="", help="Postfix for the output directory")
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    benchmark_dir = args.benchmark_dir
    
    for verifier in os.listdir(args.output_dir):
        data = []
        for benchmark_type in os.listdir(os.path.join(args.output_dir, verifier)):   
            timeout = 30 if benchmark_type == 'time_invariant' else 60.0
            for benchmark in os.listdir(os.path.join(benchmark_dir, benchmark_type)):
                output_dir_path = os.path.join(args.output_dir, verifier, benchmark_type, benchmark)
                instances_file = os.path.join(benchmark_dir, benchmark_type, benchmark, "instances.csv")
                print(benchmark_type, benchmark)
                with open(instances_file, "r") as f:
                    instances = f.readlines()
                for instance in instances:
                    onnx, vnnlib, _ = instance.strip().split(",")
                    onnx_path = os.path.join(benchmark_dir, benchmark_type, benchmark, onnx)
                    vnnlib_path = os.path.join(benchmark_dir, benchmark_type, benchmark, vnnlib)
                    
                    assert os.path.exists(onnx_path), f"Does not exist: {onnx_path=}"
                    assert os.path.exists(vnnlib_path), f"Does not exist: {vnnlib_path=}"
                    
                    output_path = os.path.abspath(os.path.join(output_dir_path, f'{os.path.splitext(os.path.basename(onnx))[0]}_{os.path.splitext(os.path.basename(vnnlib))[0]}'))
                    result_path = f'{output_path}.txt'
                    
                    if benchmark_type == 'time_invariant':
                        if benchmark.startswith('image_'):
                            # onnx/0_cifar100_0.1_cifar100_motion_blur_0_5.onnx
                            # print(onnx)
                            parts = os.path.splitext(os.path.basename(onnx))[0].split('_')
                            task = 'image'
                            perturbation_type = f'motion_blur_{parts[-2]}'
                            kernel_size = int(parts[-1])
                            # strength = float(parts[-6])
                            n_channel = '-'
                            model_name = benchmark.replace('image_', '')
                            strength = float(onnx.split(f'_{model_name}_')[1])
                            # print(f'{task=},{model_name=},{n_channel=},{kernel_size=},{perturbation_type=},{strength=}')
                            # exit()
                        else:
                            _, _, task, model_name, n_channel, kernel_size, perturbation_type, strength = os.path.splitext(os.path.basename(onnx))[0].split('_')
                    else:
                        _, _, task, model_name, n_channel, perturbation_type, kernel_size, strength = os.path.splitext(os.path.basename(onnx))[0].split('_')
                        
                    if not os.path.exists(result_path):
                        status, runtime = 'timeout', timeout
                    else:
                        status, runtime = open(result_path).read().strip().split(',')
                        if float(runtime) > timeout:
                            status = 'timeout'
                            runtime = timeout
                        
                    # print(f'{task},{model_name},{n_channel},{kernel_size},{perturbation_type},{strength},{status},{runtime}')
                    data.append([benchmark_type, task, model_name, n_channel, kernel_size, perturbation_type, strength, status, runtime])
                    
        df = pd.DataFrame(data, columns=['benchmark_type', 'task', 'model_name', 'n_channel', 'kernel_size', 'perturbation_type', 'strength', 'status', 'runtime'])
        if args.postfix:
            df.to_csv(os.path.join(os.path.dirname(__file__), f'{verifier}_results_{args.postfix}.csv'), index=False)
        else:
            df.to_csv(os.path.join(os.path.dirname(__file__), f'{verifier}_results.csv'), index=False)
                        
if __name__ == "__main__":
    main()