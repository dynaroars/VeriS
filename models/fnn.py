import torch.nn.functional as F
import torch.nn as nn
import torch

class F2(nn.Module):
    def __init__(self, input_size: int = 784, hidden_size: int = 256, output_size: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 28, 28]
        x = self.layers(x)  
        return x # x: [B, 10]
    
class F4(nn.Module):
    def __init__(self, input_size: int = 784, hidden_size: int = 256, output_size: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 28, 28]
        x = self.layers(x)  
        return x # x: [B, 10]   
    