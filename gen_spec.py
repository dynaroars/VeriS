from torch.utils.data import DataLoader
import warnings
import argparse
import torch
import os

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TorchScript-based ONNX export.*")

from utils import set_seed, get_device, load_checkpoint, evaluate_model, get_model, get_datasets, get_checkpoint_path
from specifications.spec_time_invariant import generate_time_invariant_spec
from specifications.spec_time_varying import generate_time_varying_spec
from specifications.spec_rotate import generate_rotate_spec

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", type=str, required=True, choices=["kws", "ecg", "geometric"])
    p.add_argument("--model", type=str, default="m5", choices=["m5", "m3", "f2", "f4"])
    p.add_argument("--n_channel", type=int, default=32)
    p.add_argument("--sample_per_class", type=int, default=1)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    p.add_argument("--data_dir", type=str, default="./data/", help="Root directory for data")
    p.add_argument("--spec_dir", type=str, default="./generated_benchmark/", help="Root directory for specs")
    args = p.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.spec_dir, exist_ok=True)
    return args

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
    
    if args.task == "geometric":
        generate_rotate_spec(args, model, test_loader, checkpoint["label_to_index"], device)
    else:
        generate_time_invariant_spec(args, model, test_loader, checkpoint["label_to_index"], device)
        generate_time_varying_spec(args, model, test_loader, checkpoint["label_to_index"], device)
    
if __name__ == "__main__":
    main()