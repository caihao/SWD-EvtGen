import torch
from torch import nn
from torch.nn import Linear, LayerNorm, GELU, ReLU, Module, Identity


class LinearEmbedding(Module):
    def __init__(self, latent_num, quantities_num, embedding_num, norm='ln', activation='gelu'):
        super().__init__()
        self.fc = Linear(latent_num, quantities_num * embedding_num)
        self.quantities_num = quantities_num
        self.embedding_num = embedding_num
        assert norm in [None, 'ln']
        if norm is None:
            self.norm = Identity()
        elif norm == 'ln':
            self.norm = LayerNorm(embedding_num)
        assert activation in ['relu', 'gelu']
        if activation == 'relu':
            self.activation = ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = GELU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.quantities_num, self.embedding_num)
        x = self.norm(x)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    latent_dim = 100
    quantities_num = 8
    embedding_num = 1024
    x = torch.randn(1, latent_dim)
    embedding_obj_list = [LinearEmbedding(latent_dim, quantities_num, embedding_num)]
    for embedding_obj in embedding_obj_list:
        print(embedding_obj(x))