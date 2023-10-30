import torch
from torch.nn import Linear, GELU, LayerNorm, Module


class LinearPredictor(Module):
    def __init__(self, embedding_num):
        super().__init__()
        self.fc = Linear(embedding_num, embedding_num)

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)
        return x.transpose(2, 1)
