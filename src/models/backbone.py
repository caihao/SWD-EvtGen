from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Parameter, Linear, Module, ReLU, GELU


class TransformerBlock(Module):
    def __init__(self, d_model, nhead, dff, activation, num_layers):
        super().__init__()
        assert activation in ['relu', 'gelu']
        if activation == 'relu':
            self.activation = ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = GELU()
        self.transformer_block = TransformerEncoder(
            TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dff, activation=activation,
                                    batch_first=True, dropout=0.), num_layers=num_layers)

    def forward(self, x):
        return self.transformer_block(x)
