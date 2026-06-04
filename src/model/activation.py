import torch
import torch.nn as nn

class CauchyActivation(nn.Module):
    """
    Cauchy Activation Function: f(x) = λ1·x/(x²+d²+ε) + λ2/(x²+d²+ε)

    ablation_mode:
      'full'         — both terms trainable (default)
      'no_even'      — λ2 fixed to 0; only odd term λ1·x/(x²+d²)
      'no_odd'       — λ1 fixed to 0; only even term λ2/(x²+d²)
      'fixed_params' — λ1, λ2, d all fixed (non-trainable)
    """
    def __init__(self, lambda1=0.5, lambda2=0.0, d=0.5, eps=1e-8, ablation_mode='full'):
        super().__init__()
        self.eps = eps
        self.ablation_mode = ablation_mode

        if ablation_mode == 'full':
            self.lambda1 = nn.Parameter(torch.tensor(float(lambda1)))
            self.lambda2 = nn.Parameter(torch.tensor(float(lambda2)))
            self.d = nn.Parameter(torch.tensor(float(d)))
            with torch.no_grad():
                nn.init.normal_(self.lambda1, mean=float(lambda1), std=0.1)
                nn.init.normal_(self.lambda2, mean=float(lambda2), std=0.1)
                nn.init.normal_(self.d, mean=float(d), std=0.1)
        elif ablation_mode == 'no_even':
            self.lambda1 = nn.Parameter(torch.tensor(float(lambda1)))
            self.d = nn.Parameter(torch.tensor(float(d)))
            self.register_buffer('lambda2', torch.tensor(0.0))
            with torch.no_grad():
                nn.init.normal_(self.lambda1, mean=float(lambda1), std=0.1)
                nn.init.normal_(self.d, mean=float(d), std=0.1)
        elif ablation_mode == 'no_odd':
            self.lambda2 = nn.Parameter(torch.tensor(float(lambda2)))
            self.d = nn.Parameter(torch.tensor(float(d)))
            self.register_buffer('lambda1', torch.tensor(0.0))
            with torch.no_grad():
                nn.init.normal_(self.lambda2, mean=float(lambda2), std=0.1)
                nn.init.normal_(self.d, mean=float(d), std=0.1)
        elif ablation_mode == 'fixed_params':
            self.register_buffer('lambda1', torch.tensor(float(lambda1)))
            self.register_buffer('lambda2', torch.tensor(float(lambda2)))
            self.register_buffer('d', torch.tensor(float(d)))
        else:
            raise ValueError(f"Unknown ablation_mode: '{ablation_mode}'")

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