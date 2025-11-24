import torch.nn.functional as F
import torch.nn as nn
import torch

class M3(nn.Module):

    def __init__(self, n_input: int = 1, n_output: int = 35, 
                 stride: int = 16, kernel_size: int = 80, n_channel: int = 32, 
                 length: int = 16000):
        super().__init__()

        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=(1, kernel_size), stride=(stride, stride))
        self.bn1 = nn.BatchNorm2d(n_channel, momentum=0.1)
        self.pool1 = nn.Conv2d(n_channel, n_channel, kernel_size=(1, 4), stride=(4, 4), groups=n_channel, bias=False)

        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=(1, 3))
        self.bn2 = nn.BatchNorm2d(n_channel, momentum=0.1)
        self.pool2 = nn.Conv2d(n_channel, n_channel, kernel_size=(1, 4), stride=(4, 4), groups=n_channel, bias=False)

        self.conv3 = nn.Conv2d(n_channel, n_channel, kernel_size=(1, 3))
        self.bn3 = nn.BatchNorm2d(n_channel, momentum=0.1)
        self.pool3 = nn.Conv2d(n_channel, n_channel, kernel_size=(1, 4), stride=(4, 4), groups=n_channel, bias=False)

        self.fc1 = nn.Linear(n_channel, n_output)
        self.act = torch.tanh
        self.length = length
            
        self._init_weights_xavier()
        self._init_pooling_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        x = x.view(-1, 1, 1, self.length)  # [B, 1, 1, T] 
        x = self.conv1(x)
        x = self.act(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act(self.bn2(x))
        x = self.pool2(x)
        
        # x = self.conv3(x)
        # x = self.act(self.bn3(x))
        # x = self.pool3(x)
        
        x = x.mean(dim=(2, 3))  # [B, C, H, W] -> [B, C]
        x = self.fc1(x)    # [B, n_output]
        return x
    
    def _init_weights_xavier(self):
        """Initialize weights using Xavier/Glorot initialization for sigmoid/tanh"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                m.weight.data *= 0.5  # Reduce scale, helps with saturation
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.1)  # Less squashy at start
                nn.init.constant_(m.bias, 0)
                
    def _init_pooling_layers(self):
        """Initialize Conv2d pooling layers to act as average pooling"""
        with torch.no_grad():
            for pool_layer in [self.pool1, self.pool2, self.pool3]:
                # Set weights to 1/kernel_size for average pooling behavior (1x4 kernels)
                pool_layer.weight.fill_(1.0 / pool_layer.kernel_size[1])  # Only width dimension matters


if __name__ == "__main__":
    stride =8
    
    model = M3(length=2714, stride=stride)
    x = torch.randn(1, 1, 2714)
    print(model(x).shape)
    
    model = M3(length=4000, stride=stride)
    x = torch.randn(1, 1, 4000)
    print(model(x).shape)