import torch
import torch.nn as nn

from src.model.activation import CauchyActivation


DEFAULT_CAUCHY_PARAMS = {"lambda1": 0.7, "lambda2": 0.1, "d": 0.5}


def _cauchy_params_or_default(cauchy_params):
    return dict(cauchy_params or DEFAULT_CAUCHY_PARAMS)


def _init_linear(layer, gain=1.234):
    nn.init.xavier_normal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class TunedXNet(nn.Module):
    """XNet variant used by the multi-model tuning experiment."""

    def __init__(self, feature_size, hidden_dim=128, num_layers=3, cauchy_params=None):
        super().__init__()
        cauchy_params = _cauchy_params_or_default(cauchy_params)
        self.act = CauchyActivation(**cauchy_params)
        self.layers = nn.ModuleList([nn.Linear(feature_size, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)

        for layer in list(self.layers) + [self.out]:
            _init_linear(layer)

    def forward(self, x):
        x = x.squeeze(-1)
        for layer in self.layers:
            x = self.act(layer(x))
        return self.out(x)


class FCModel(nn.Module):
    """Fully connected Cauchy baseline."""

    def __init__(self, feature_size, hidden_dim=64, num_layers=2, cauchy_params=None):
        super().__init__()
        cauchy_params = _cauchy_params_or_default(cauchy_params)
        self.act = CauchyActivation(**cauchy_params)

        layers = [nn.Linear(feature_size, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_dim, 1)

        for layer in list(self.layers) + [self.out]:
            _init_linear(layer)

    def forward(self, x):
        x = x.squeeze(-1)
        for layer in self.layers:
            x = self.act(layer(x))
        return self.out(x)


class LSTMModel(nn.Module):
    """Single-step sequence baseline using a Cauchy-transformed output."""

    def __init__(self, feature_size, hidden_dim=64, num_layers=2, cauchy_params=None):
        super().__init__()
        cauchy_params = _cauchy_params_or_default(cauchy_params)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.act = CauchyActivation(**cauchy_params)
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.out = nn.Linear(hidden_dim, 1)
        self._init_recurrent(self.lstm)
        _init_linear(self.out)

    @staticmethod
    def _init_recurrent(module):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=1.234)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_dim)
        _, (hidden, _) = self.lstm(x, (h0, c0))
        return self.act(self.out(hidden[-1]))


class GRUModel(nn.Module):
    """GRU baseline using a Cauchy-transformed output."""

    def __init__(self, feature_size, hidden_dim=64, num_layers=2, cauchy_params=None):
        super().__init__()
        cauchy_params = _cauchy_params_or_default(cauchy_params)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.act = CauchyActivation(**cauchy_params)
        self.gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.out = nn.Linear(hidden_dim, 1)
        LSTMModel._init_recurrent(self.gru)
        _init_linear(self.out)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_dim)
        _, hidden = self.gru(x, h0)
        return self.act(self.out(hidden[-1]))


class CNNModel(nn.Module):
    """1D convolutional baseline for sliding-window features."""

    def __init__(self, feature_size, hidden_dim=64, num_layers=2, cauchy_params=None):
        super().__init__()
        cauchy_params = _cauchy_params_or_default(cauchy_params)
        self.act = CauchyActivation(**cauchy_params)
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

        for layer in [self.cnn1, self.cnn2]:
            nn.init.xavier_normal_(layer.weight, gain=1.234)
            nn.init.zeros_(layer.bias)
        _init_linear(self.fc)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.act(self.cnn1(x))
        x = self.act(self.cnn2(x))
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


class ResBlock(nn.Module):
    def __init__(self, dim, cauchy_params):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = CauchyActivation(**cauchy_params)
        _init_linear(self.linear1)
        _init_linear(self.linear2)

    def forward(self, x):
        residual = x
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return self.act(x + residual)


class ResNetModel(nn.Module):
    """Residual fully connected Cauchy baseline."""

    def __init__(self, feature_size, hidden_dim=64, num_layers=2, cauchy_params=None):
        super().__init__()
        cauchy_params = _cauchy_params_or_default(cauchy_params)
        self.act = CauchyActivation(**cauchy_params)
        self.input_proj = nn.Linear(feature_size, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_dim, cauchy_params) for _ in range(num_layers)]
        )
        self.out = nn.Linear(hidden_dim, 1)
        _init_linear(self.input_proj)
        _init_linear(self.out)

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.act(self.input_proj(x))
        for block in self.res_blocks:
            x = block(x)
        return self.act(self.out(x))


MODEL_REGISTRY = {
    "XNet": TunedXNet,
    "FC": FCModel,
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "CNN": CNNModel,
    "ResNet": ResNetModel,
}


def build_model(model_name, feature_size, hidden_dim=64, num_layers=2, cauchy_params=None):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(feature_size, hidden_dim, num_layers, cauchy_params)
