import torch
import torch.nn as nn
from src.model.activation import CauchyActivation, StandardActivation

class XNet(nn.Module):
    def __init__(self, feature_size, hidden_dim=64, num_layers=2, activation='cauchy', cauchy_params=None):
        super().__init__()
        # Build layers
        layers = [nn.Linear(feature_size, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_dim, 1)

        # Activation function
        if activation == 'cauchy':
            self.act = CauchyActivation(**(cauchy_params if cauchy_params else {}))
        else:
            self.act = StandardActivation(activation)

        # Weight initialization
        for m in list(self.layers) + [self.out]:
            if isinstance(m, nn.Linear):
                if activation == 'cauchy':
                    nn.init.xavier_normal_(m.weight, gain=1.234)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.squeeze(-1)
        for lin in self.layers:
            x = lin(x)
            x = self.act(x)
        x = self.out(x)
        return x