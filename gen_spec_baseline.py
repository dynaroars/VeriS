from torch.utils.data import DataLoader
import warnings
import argparse
import torch
import os

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TorchScript-based ONNX export.*")

from utils import set_seed, get_device, load_checkpoint, evaluate_model, create_vnnlib_str, get_model, get_datasets, get_checkpoint_path, get_valid_data
from perturbations.min_max_kernel import find_neighborhood_bounds
from perturbations.time_invariant import get_kernel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", type=str, required=True, choices=["kws", "ecg"])
    p.add_argument("--model", type=str, default="m5", choices=["m5", "m3"])
    p.add_argument("--n_channel", type=int, default=32)
    p.add_argument("--sample_per_class", type=int, default=1)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    p.add_argument("--data_dir", type=str, default="./data/", help="Root directory for data")
    p.add_argument("--spec_dir", type=str, default="./generated_benchmark_baseline/", help="Root directory for specs")
    args = p.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.spec_dir, exist_ok=True)
    return args


def generate_time_invariant_spec(args, model, test_loader, label_to_index, device):
    print(f'\n{"="*80}')
    print(f"Starting {args.task} {args.model} Time-Invariant Baseline specs generation...")
    print(f'{"="*80}\n')
    
    count = 0
    valid_data = get_valid_data(args, model, test_loader, label_to_index, device)
    
    spec_dir = os.path.join(args.spec_dir, f'time_invariant', f'{args.task}_{args.model}_{args.n_channel}')
    os.makedirs(os.path.join(spec_dir, 'vnnlib'), exist_ok=True)
    os.makedirs(os.path.join(spec_dir, 'onnx'), exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)
    
    if args.task == "kws":
        kernel_sizes = [301, 501, 701]
    elif args.task == "ecg":
        kernel_sizes = [51, 101, 151]
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    with open(os.path.join(spec_dir, f'instances.csv'), 'w') as f, open(os.path.join(spec_dir, f'command.sh'), 'w') as f2:
        for kernel_size in kernel_sizes:
            for perturbation_type in ['lowpass', 'echo', 'highpass']:
                for strength in [0.1, 0.5, 1.0]:
                    for x, y, logit in valid_data:
                        base_name = f"{count}_{args.seed}_{args.task}_{args.model}_{args.n_channel}_{kernel_size}_{perturbation_type}_{strength}"
                        onnx_name = os.path.join('onnx', f"{base_name}.onnx")

                        # spec
                        kernel = get_kernel(perturbation_type, kernel_size)
                        k_max = kernel.abs().max() * strength
                        # x_ub, x_lb = find_neighborhood_bounds_old(x.flatten().cpu().numpy(), kernel, kernel_bounds=(-k_max, k_max))
                        x_ub, x_lb = find_neighborhood_bounds(x.flatten().cpu().numpy(), kernel, kernel_bounds=(-k_max, k_max))
                        x_lb = x_lb.reshape(x.shape)
                        x_ub = x_ub.reshape(x.shape)
                        assert (x_lb <= x_ub).all()
                        
                        specs = create_vnnlib_str(
                            data_lb=x_lb, 
                            data_ub=x_ub, 
                            prediction=logit,
                        )
                        
                        for i, spec in enumerate(specs):
                            spec_name = os.path.join('vnnlib', f"{base_name}_{i}.vnnlib")
                                
                            with open(os.path.join(spec_dir, spec_name), 'w') as fs:
                                print(spec, file=fs)
                            print(f'{onnx_name},{spec_name},{args.timeout}', file=f)
                            
                            # command
                            print(f'python3 main.py --net {os.path.abspath(spec_dir)}/onnx/{base_name}.onnx --spec {os.path.abspath(spec_dir)}/vnnlib/{base_name}_{i}.vnnlib --timeout {args.timeout}', file=f2)
                            
                        model.cpu()
                        model.eval()
                                  
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            torch.onnx.export(
                                model,
                                x_lb,
                                os.path.join(spec_dir, onnx_name),
                                opset_version=12,
                                input_names=["input"],
                                output_names=["output"],
                                dynamic_axes={
                                    'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'},
                                }
                            )
                        
                        count += 1
                        
                        if count % 100 == 0:
                            print(f"[!] Generated {count * len(specs)} specs")

            #             break
            #         break
            #     break
            # break
                        
    print(f"[!] Time invariant: {count * len(specs)} specs")
            
@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = get_device()
    print(f"Using {device=}")
    
    _, _, test_ds, _, num_classes = get_datasets(args)
    test_loader = DataLoader(
        test_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=os.cpu_count(), 
        pin_memory=True, 
        drop_last=False,
    )
    print(f'Dataloaders: {len(test_ds)=}')
    
    model = get_model(args, num_classes)
    print(model)
    model.to(device)
    
    # load checkpoint
    checkpoint_path = get_checkpoint_path(args)
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # evaluate model
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")
    
    generate_time_invariant_spec(args, model, test_loader, checkpoint["label_to_index"], device)

if __name__ == "__main__":
    main()