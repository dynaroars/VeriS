import argparse
import tqdm
import os

from verifier import neuralsat, abcrown, abcrown_A
from utils import recursive_walk

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_dir", type=str, default="generated_benchmark/", help="Root directory for benchmark")
    p.add_argument("--benchmark_type", type=str, required=True, choices=["time_invariant", "time_varying", "geometric"])
    p.add_argument("--verifier", type=str, required=True, choices=["neuralsat", "abcrown", "abcrown_A"])
    p.add_argument("--verifier_dir", type=str, required=True, help="Verifier directory")
    p.add_argument("--home_dir", type=str, default=os.getcwd(), help="Home directory for verifier")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory")
    p.add_argument("--timeout", type=int, default=30, help="Timeout")
    args = p.parse_args()
    args.verifier_dir = os.path.abspath(os.path.join(args.home_dir, args.verifier_dir))
    return args

def get_total_instances(benchmark_dir):
    count = 0
    for file in tqdm.tqdm(recursive_walk(benchmark_dir)):
        if file.endswith('instances.csv'):
            count += len(open(file).readlines())
    return count

def main():
    args = parse_args()
    benchmark_dir = os.path.join(args.benchmark_dir, args.benchmark_type)
    assert os.path.exists(benchmark_dir), f"Benchmark directory does not exist: {benchmark_dir=}"
    
    if args.verifier == "neuralsat":
        verify_func = neuralsat.verify
    elif args.verifier == "abcrown":
        verify_func = abcrown.verify
    elif args.verifier == "abcrown_A":
        verify_func = abcrown_A.verify
    else:
        raise ValueError(f"Invalid verifier: {args.verifier=}")
    
    total_instances = get_total_instances(benchmark_dir)
    
    pbar = tqdm.tqdm(total=total_instances)
    
    stats = {
        'sat': 0,
        'unsat': 0,
        'timeout': 0,
        'error': 0,
    }
    
    print(f'[+] Running {args.verifier=} {args.benchmark_type=}')
    
    for benchmark in os.listdir(benchmark_dir):
        pbar.set_description(f'{benchmark=}')
        
        output_dir = os.path.join(args.output_dir, args.verifier, args.benchmark_type, benchmark)
        os.makedirs(output_dir, exist_ok=True)
        
        instances_file = os.path.join(benchmark_dir, benchmark, 'instances.csv')
        assert os.path.exists(instances_file), f"Instances file does not exist: {instances_file=}"
        
        with open(instances_file, 'r') as f:
            instances = f.readlines()
        
        for instance in instances:
            onnx, vnnlib, _ = instance.strip().split(',')
            onnx_path = os.path.abspath(os.path.join(benchmark_dir, benchmark, onnx))
            vnnlib_path = os.path.abspath(os.path.join(benchmark_dir, benchmark, vnnlib))
            output_path = os.path.abspath(os.path.join(output_dir, f'{os.path.splitext(os.path.basename(onnx))[0]}_{os.path.splitext(os.path.basename(vnnlib))[0]}'))
            
            # timeout = float(timeout)
            assert os.path.exists(onnx_path), f"Does not exist: {onnx_path=}"
            assert os.path.exists(vnnlib_path), f"Does not exist: {vnnlib_path=}"
            # print(f'[+] Running {onnx_path=} {vnnlib_path=} {timeout=}')
            status = verify_func(args, onnx_path, vnnlib_path, output_path, args.timeout)
            stats[status] += 1
            pbar.update(1)
            pbar.set_postfix(**stats)
        
if __name__ == "__main__":
    main()