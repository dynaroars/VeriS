import warnings
import torch
import os

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TorchScript-based ONNX export.*")

from perturbations.time_invariant import TimeInvariantPerturbationLayer
from utils import create_vnnlib_str, get_valid_data

def generate_time_invariant_spec(args, model, test_loader, label_to_index, device):
    print(f'\n{"="*80}')
    print(f"Starting {args.task} {args.model} Time-Invariant specs generation...")
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
                        x_lb = torch.tensor([[0.0]])
                        x_ub = torch.tensor([[strength]])
                        
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
                            
                            
                        # network
                        perturbed_layer = TimeInvariantPerturbationLayer(
                            input_signal=x.squeeze(1), 
                            perturbation_type=perturbation_type, 
                            kernel_size=kernel_size,
                        ).cpu()
                        
                        # print(perturbed_layer(x_lb).shape)
                        # exit()
                        
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
            
           