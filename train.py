from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import warnings
import argparse
import os

warnings.filterwarnings("ignore")

from utils import set_seed, get_device, save_checkpoint, get_model_parameters, evaluate_model, get_model, get_datasets
from engine.trainer import Trainer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["kws", "ecg"])
    p.add_argument("--model", type=str, default="m5", choices=["m5", "m3"])
    p.add_argument("--n_channel", type=int, default=32)
    p.add_argument("--data_dir", type=str, default="./data", help="Root directory for data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    args = p.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    return args

def main():
    args = parse_args()
    set_seed(args.seed)
    
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.task}_{args.model}_{args.n_channel}.pt")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}. Done.")
        return
    
    device = get_device()
    print(f"Using {device=}")

    train_ds, val_ds, test_ds, label_mapping, num_classes = get_datasets(args)
    
    # Dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=False,
    )
    print(f'Dataloaders: {len(train_ds)=} {len(val_ds)=} {len(test_ds)=}')
    
    # Model
    model = get_model(args, num_classes)
    print(model)
    print(f"Model parameters: {get_model_parameters(model)}")
    model.to(device)

    # Optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Add gradient clipping for sigmoid/tanh activations
    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device, 
        scheduler=None, 
        use_amp=True,
        max_grad_norm=None
    )

    print(f"Starting {args.task} training...")
    trainer.fit(train_loader, val_loader, epochs=args.epochs, log_interval=1)

    # Final test evaluation (same for both tasks)
    print("Evaluating on test set...")
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save checkpoint (same for both tasks)
    checkpoint_data = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "label_to_index": label_mapping,
    }
    
    save_checkpoint(checkpoint_path, checkpoint_data)
    print(f"Saved checkpoint to {checkpoint_path}\n")

if __name__ == "__main__":
    main()