import torch
import torch.nn as nn

class CauchyActivation(nn.Module):
    def __init__(self, lambda1=0.5, lambda2=0.0, d=0.5, eps=1e-8):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.tensor(float(lambda1)))
        self.lambda2 = nn.Parameter(torch.tensor(float(lambda2)))
        self.d = nn.Parameter(torch.tensor(float(d)))
        self.eps = eps
        
        # Initialization
        with torch.no_grad():
            nn.init.normal_(self.d, mean=float(d), std=0.1)
            nn.init.normal_(self.lambda1, mean=float(lambda1), std=0.1)
            nn.init.normal_(self.lambda2, mean=float(lambda2), std=0.1)

    def forward(self, x):
        denominator = x.pow(2) + self.d.pow(2) + self.eps
        return self.lambda1 * x / denominator + self.lambda2 / denominator

class StandardActivation(nn.Module):
    def __init__(self, activation_name):
        super().__init__()
        if activation_name == 'relu':
            self.act = nn.ReLU()
        elif activation_name == 'tanh':
            self.act = nn.Tanh()
        elif activation_name == 'gelu':
            self.act = nn.GELU()
        elif activation_name == 'leaky_relu':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def forward(self, x):
        return self.act(x)