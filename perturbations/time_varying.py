import torch.nn as nn
import numpy as np
import torch

class TimeVaryingPerturbationLayer(nn.Module):

    def __init__(self, input_signal, displacement_type='linear', max_displacement=2.0, window_size=None):
        super().__init__()
        self.x = input_signal.clone()  # [1, T]
        self.T = self.x.shape[1]
        self.displacement_type = displacement_type
        self.max_displacement = max_displacement
        self.window_size = window_size if window_size is not None else self.T
        self.n = torch.arange(self.T, dtype=torch.float32).view(1, self.T)
        self.k = torch.arange(self.T, dtype=torch.float32).view(1, 1, self.T)
        self.coeffs = self._extract_displacement_coefficients()
        self.create_s_layer()
        
    def create_s_layer(self):
        # Linear layer for computing s = w * coeffs + n
        self.s_layer = nn.Linear(1, self.T, bias=True)
        with torch.no_grad():
            # Weight: coeffs transposed to match linear layer format [T, 1]
            self.s_layer.weight.copy_(self.coeffs.T)
            self.s_layer.weight.requires_grad_(False)
            # Bias: n
            self.s_layer.bias.copy_(self.n.squeeze(0))
            self.s_layer.bias.requires_grad_(False)
            
        self.warped_layer = nn.Linear(self.x.shape[1], 1, bias=False)
        with torch.no_grad():
            self.warped_layer.weight.copy_(self.x)   # signal shape [1, 64]
            self.warped_layer.weight.requires_grad_(False)  # keep it fixed
        
    
    def _extract_displacement_coefficients(self):
        # Create window indices for the base pattern
        window_n = torch.arange(self.window_size, dtype=torch.float32)
        
        if self.displacement_type == 'linear':
            # u[n] = w * max_displacement * (n/window_size - 0.5)
            window_coeffs = self.max_displacement * (window_n / self.window_size - 0.5)
        elif self.displacement_type == 'sinusoidal':
            # u[n] = w * max_displacement * sin(2π * n / window_size)
            window_coeffs = self.max_displacement * torch.sin(2 * torch.pi * window_n / self.window_size)
        elif self.displacement_type == 'quadratic':
            # u[n] = w * max_displacement * ((n/window_size - 0.5)^2 - 0.25)
            normalized_n = window_n / self.window_size - 0.5
            window_coeffs = self.max_displacement * (normalized_n**2 - 0.25)
        elif self.displacement_type == 'gaussian':
            # u[n] = w * max_displacement * exp(-(n/window_size - 0.5)^2 / (2 * sigma^2))
            normalized_n = window_n / self.window_size - 0.5
            sigma = 0.2
            window_coeffs = self.max_displacement * torch.exp(-(normalized_n**2) / (2 * sigma**2))
        else:
            raise ValueError(f"Unknown displacement type: {self.displacement_type}")
        
        # Repeat the window pattern to fill T samples
        num_full_cycles = self.T // self.window_size
        remainder = self.T % self.window_size
        
        coeffs_list = []
        # Add full cycles
        for _ in range(num_full_cycles):
            coeffs_list.append(window_coeffs)
        
        # Add partial cycle if needed
        if remainder > 0:
            coeffs_list.append(window_coeffs[:remainder])
        
        # Concatenate and reshape to [1, T]
        coeffs = torch.cat(coeffs_list, dim=0).view(1, self.T)
        
        return coeffs
    
    def apply_time_warp_unoptimized(self, w):
        # compute s = n + u using linear layer
        # s1 = self.s_layer(w)  # [B, T]
        s = w * self.coeffs + self.n
        # assert torch.allclose(s, s1, atol=1e-6)
        # d = s - k
        d = s.unsqueeze(2) - self.k # [B, T, T]
        # P = ψ(s - k), where ψ(x) = ReLU(1 - |x|)
        # P  = torch.relu(1.0 - torch.abs(d))
        # no abs 
        # P1 = torch.relu(1.0 - torch.relu(d) - torch.relu(-d))  # ψ(d) = ReLU(1 - ReLU(d) - ReLU(-d)) # [B, T, T]
        P  = torch.relu(1.0 - torch.abs(d))
        # assert torch.allclose(P, P1, atol=1e-6), f'{torch.norm(P - P1)=}'
        
        # Weighted sum over k
        warped = torch.sum(P * self.x.unsqueeze(1), dim=-1)  # [B, T]
        # assert torch.allclose(warped, warped2, atol=1e-6), f'{torch.norm(warped - warped2)=}'
        # print(f"{warped2.shape=}, {warped.shape=}")
        return warped

    
    
    def apply_time_warp(self, w):
        # compute s = n + u using linear layer
        s = self.s_layer(w)  # [B, T]
        # d = s - k
        d = s.unsqueeze(2) - self.k # [B, T, T]
        # P = ψ(s - k), where ψ(x) = ReLU(1 - |x|)
        # P  = torch.relu(1.0 - torch.abs(d))
        # no abs 
        P = torch.relu(1.0 - torch.relu(d) - torch.relu(-d))  # ψ(d) = ReLU(1 - ReLU(d) - ReLU(-d)) # [B, T, T]
        # print(f'{P=}')
        # Weighted sum over k
        warped = self.warped_layer(P)
        return warped

    
    
    def get_displacement_field(self, w):
        # Compute displacement: u(w) = w * coeffs -> [batch_size, T]
        # assert len(w.shape) == 2, f"w must be a 2D tensor, got {w.shape=}"
        return w * self.coeffs
    
    def get_warped_signal(self, w):
        # w1 = self.apply_time_warp_unoptimized(w)
        # return w1
        w2 = self.apply_time_warp(w)
        # print(f"{w1.shape=}, {w2.shape=}")
        # assert torch.allclose(w1, w2.flatten(1), atol=1e-3), f'{torch.norm(w1 - w2.flatten(1))=}'
        return w2
    
    
    def forward(self, w):
        # z, w = z_w.split(1, dim=1)
        return self.get_warped_signal(w) # [B, 1, T]


@torch.no_grad()
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    T = 4000
    x = torch.randn(1, T)
    
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    T = x.shape[1]
    
    Pz = TimeVaryingPerturbationLayer(x, displacement_type='sinusoidal', max_displacement=2.0, window_size=T)
    # layer = TimeVaryingPerturbationLayer(x, displacement_type='linear', max_displacement=2.0, window_size=32)
    # # Test batched inputs
    # z_batch = torch.tensor([[0.1], [0.5], [1.0], [1.5], [2.0], [2.5]])  # [B, 1]
    # w_batch = torch.tensor([[0.5], [1.0], [1.5]])  # [B, 1]
    z = torch.tensor([[1.0]])
    # print(f"{z_batch.shape=}, {w_batch.shape=}")
    # zw = torch.cat([z_batch, w_batch], dim=1) # [B, 2]
    
    print(f"{x=}")
    print(f"{Pz.coeffs.int()=}")
    
    
    print(f'{Pz(z)=}')
    # print(f"{x.shape=}, {z.shape=}, {y_batch.shape=}")
    
    # torch.onnx.export(layer, z, "example_time_varying.onnx", verbose=True, opset_version=12)
    
    

if __name__ == "__main__":
    main()
