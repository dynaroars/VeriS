import time
import os

def verify(args, onnx_path, vnnlib_path, output_path, timeout):
    # print(f'{result_path=}')
    result_path = f'{output_path}.txt'
    log_path = f'{output_path}.log'
    if os.path.exists(result_path):
        status = open(result_path).read().strip().split(',')[0]
        if status not in ['sat', 'unsat', 'timeout']:
            status = 'error'
        return status
    
    os.chdir(args.verifier_dir)
    
    if args.benchmark_type == 'time_invariant':
        setting_path = os.path.join(args.home_dir, 'verifier/config/neuralsat/time_invariant.json')
    elif args.benchmark_type == 'time_varying':
        setting_path = os.path.join(args.home_dir, 'verifier/config/neuralsat/time_varying.json')
    elif args.benchmark_type == 'geometric':
        setting_path = os.path.join(args.home_dir, 'verifier/config/neuralsat/geometric.json')
    else:
        raise ValueError(f'Invalid benchmark type: {args.benchmark_type=}')
    
    assert os.path.exists(setting_path), f"Setting file does not exist: {setting_path=}"
    
    cmd  = f'python3 -W ignore main.py --verbosity=2'
    cmd += f' --net {onnx_path} --spec {vnnlib_path} --timeout {timeout}'
    cmd += f' --result_file {result_path}'
    cmd += f' --setting_file {setting_path}'
    cmd += f' --export_runtime'
    cmd += f' > {log_path} 2>&1'

    tic = time.time()
    os.system(cmd)
    toc = time.time()
    
    if os.path.exists(result_path):
        status = open(result_path).read().strip().split(',')[0]
        if status not in ['sat', 'unsat', 'timeout']:
            status = 'error'
    else:
        status = 'error'
        with open(result_path, 'w') as f:
            print(f'{status},{toc - tic}', file=f)
    os.chdir(args.home_dir)
    return status