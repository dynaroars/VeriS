from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict, Optional
import torch
import time

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = True,
        max_grad_norm: Optional[float] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=use_amp)
        self.max_grad_norm = max_grad_norm

    def _step(self, batch, train: bool = True) -> Dict[str, float]:
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
            logits = self.model(x)              # [B, num_classes]
            loss = self.criterion(logits, y)    # CrossEntropy

        if train:
            self.scaler.scale(loss).backward()
            
            # Apply gradient clipping if specified
            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

        return {"loss": float(loss.detach().cpu().item()), "acc": acc}

    def run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        agg = defaultdict(float)
        n = 0
        for batch in loader:
            metrics = self._step(batch, train=train)
            bs = batch[0].size(0)
            for k, v in metrics.items():
                agg[k] += v * bs
            n += bs
        for k in agg:
            agg[k] /= max(n, 1)
        if (not train) and (self.scheduler is not None):
            # Step scheduler per epoch on validation if you use ReduceLROnPlateau; otherwise change as needed
            if hasattr(self.scheduler, "step") and "loss" in agg:
                try:
                    self.scheduler.step(agg["loss"])
                except TypeError:
                    self.scheduler.step()
        return dict(agg)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, log_interval: int = 1):
        best_val = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.run_epoch(train_loader, train=True)
            val_metrics = self.run_epoch(val_loader, train=False)
            dt = time.time() - t0

            if (epoch % log_interval) == 0:
                print(f"Epoch {epoch:03d} / {epochs:03d} |  "
                      f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
                      f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} | "
                      f"time={dt:.1f}s")

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_val