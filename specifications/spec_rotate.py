from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import torch
import os
import gc

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TorchScript-based ONNX export.*")

from perturbations.rotate_layer import RotationPerturbationLayer
from utils import create_vnnlib_str, get_valid_data
       
            
def _process_rotate_single_spec(args, model, spec_dir, theta, x, y, logit, count, seed):
    try:
        base_name = f"{count}_{seed}_{args.task}_{args.model}_rotate_{theta[0]}-{theta[1]}"
        onnx_name = os.path.join('onnx', f"{base_name}.onnx")
        
        x_lb = torch.tensor([[theta[0]]])
        x_ub = torch.tensor([[theta[1]]])  
        
        x_lb = torch.deg2rad(x_lb)
        x_ub = torch.deg2rad(x_ub)
        specs = create_vnnlib_str(
            data_lb=x_lb, 
            data_ub=x_ub, 
            prediction=logit,
        )        
        perturbed_layer = RotationPerturbationLayer(image=x.squeeze(0),).cpu()
        perturbed_net = torch.nn.Sequential(perturbed_layer, model).cpu()
        perturbed_net.eval()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                perturbed_net,
                x_lb,
                os.path.join(spec_dir, onnx_name),
                opset_version=12,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                },
                verbose=False
            )
        
        del x_lb, x_ub, perturbed_layer, perturbed_net
        gc.collect()
        
        res = []
        for i, spec in enumerate(specs):
            spec_name = os.path.join('vnnlib', f"{base_name}_{i}.vnnlib")
            with open(os.path.join(spec_dir, spec_name), 'w') as f:
                print(spec, file=f)
            res.append(f'{onnx_name},{spec_name},{args.timeout}')
        
        return {'success': True, 'result': res, 'task': (args, model, spec_dir, theta, x, y, logit, count, seed)}
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Failed to process {count}th spec")
        return {'success': False, 'task': (args, model, spec_dir, theta, x, y, logit, count, seed)}
         
             
def generate_rotate_spec(args, model, test_loader, label_to_index, device):

    print(f'\n{"="*80}')
    print(f"Starting {args.task} {args.model} Rotate specs generation...")
    print(f'{"="*80}\n')
    
    valid_data = get_valid_data(args, model, test_loader, label_to_index, device)
    
    spec_dir = os.path.join(args.spec_dir, f'rotate', f'{args.task}_{args.model}_{args.n_channel}')
    os.makedirs(os.path.join(spec_dir, 'vnnlib'), exist_ok=True)
    os.makedirs(os.path.join(spec_dir, 'onnx'), exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)
    
    tasks = []
    count = 0
    for theta in [
            (0.0, 15.0), (0.0, 30.0), (0.0, 45.0), 
            (15.0, 30.0), (15.0, 45.0), (15.0, 60.0), 
            (30.0, 45.0), (30.0, 60.0), (30, 90.0), 
            (45.0, 60.0), (45.0, 75.0), (45.0, 90.0),
        ]:
        for x, y, logit in valid_data:
            tasks.append((args, model, spec_dir, theta, x, y, logit, count, args.seed))
            count += 1
    
    results = Parallel(n_jobs=os.cpu_count() // 2)(
        delayed(_process_rotate_single_spec)(*task) for task in tqdm(tasks)
    )
    # results = []
    # for task in tqdm(tasks):
    #     result = _process_rotate_single_spec(*task)
    #     results.append(result)
    
    successful_results = []
    failed_results = []
    for r in results:
        if r['success']:
            successful_results.extend(r['result'])
        else:
            failed_results.append(r['task'])
    
    for failed_result in tqdm(failed_results):
        retry_result = _process_rotate_single_spec(*failed_result['task'])
        if retry_result['success']:
            successful_results.extend(retry_result['result'])
    
    with open(os.path.join(spec_dir, f'instances.csv'), 'w') as f, open(os.path.join(spec_dir, f'command.sh'), 'w') as f2:
        for result in successful_results:
            print(result, file=f)
            print(f'python3 main.py --net {os.path.abspath(spec_dir)}/{result.split(",")[0]} --spec {os.path.abspath(spec_dir)}/{result.split(",")[1]} --timeout {args.timeout}', file=f2)
    
    print(f"[!] Rotate: {len(successful_results)} specs")
    gc.collect()
    
    