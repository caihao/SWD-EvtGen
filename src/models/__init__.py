from .embedding import *
from .backbone import *
from .predictor import *
from torch import nn
from torch.nn import Module
import torch


class TransformerGenerator(Module):

    def __init__(self,
                 latent_dim: int,
                 quantities_num: int,
                 embedding_num: int,
                 nhead: int,
                 dff: int,
                 norm: str,
                 activation: 'str',
                 num_layers: int,
                 cut_theta_index=None):
        super().__init__()
        self.embedding = LinearEmbedding(latent_num=latent_dim,
                                         quantities_num=quantities_num,
                                         embedding_num=embedding_num,
                                         norm=norm,
                                         activation=activation)
        self.backbone = TransformerBlock(d_model=embedding_num,
                                         nhead=nhead,
                                         dff=dff,
                                         activation=activation,
                                         num_layers=num_layers)
        self.predictor = LinearPredictor(embedding_num=embedding_num)
        self.quantities_num = quantities_num
        self.cut_theta_index = cut_theta_index
        """
        self.model = nn.Sequential(
            LinearEmbedding(latent_num=latent_dim, quantities_num=quantities_num, embedding_num=embedding_num,
                            norm=norm,
                            activation=activation),
            TransformerBlock(d_model=embedding_num, nhead=nhead, dff=dff, activation=activation, num_layers=num_layers),
            LinearPredictor(embedding_num=embedding_num)
        )
        """

    def model_forward(self, x):
        x = self.embedding(
            x
        )  # (batch size, 100) -> (batch size, 8 * 1024) -> (batch size, 8, 1024) linear
        x = self.backbone(
            x)  # (batch size, 8, 1024) -> (batch size, 8, 1024) transformer
        x = self.predictor(x)
        return x

    def count_mass(self, momentum):
        mass = torch.sqrt(momentum[:, 3] ** 2 -
                          torch.sum(momentum[:, :3] ** 2, dim=1))
        return mass

    def forward(self,
                x,
                expand_function=None,
                free_targets=None,
                targets=None,
                criterion=None,
                return_x=False,
                return_detail=False,
                return_all=False):
        # x = self.model(inputs)
        x = self.model_forward(x)
        # assert torch.isnan(x).sum() == 0
        if free_targets is not None and targets is not None and expand_function is not None and criterion is not None:
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            free_targets = free_targets.reshape(
                free_targets.shape[0] * free_targets.shape[1],
                free_targets.shape[2])
            targets = targets.reshape(targets.shape[0] * targets.shape[1],
                                      targets.shape[2])
            x_expand = expand_function(x)
            """
            if self.cut_theta_index is not None:
                for cut_index in self.cut_theta_index:
                    index1 = torch.where(x_expand[:, cut_index] < 0.1 * torch.pi)
                    index2 = torch.where(x_expand[:, cut_index] > 0.9 * torch.pi)
                    free_targets[index1] = x[index1]
                    targets[index2] = x_expand[index2]
            """
            if self.cut_theta_index is not None:
                for cut_index in self.cut_theta_index:
                    x = x[torch.where(x_expand[:, cut_index] > 0.1 * torch.pi)]
                    x_expand = x_expand[torch.where(
                        x_expand[:, cut_index] > 0.1 * torch.pi)]
                    x = x[torch.where(x_expand[:, cut_index] < 0.9 * torch.pi)]
                    x_expand = x_expand[torch.where(
                        x_expand[:, cut_index] < 0.9 * torch.pi)]
                kppim_mass = self.count_mass(
                    x_expand[:, 24:28] + x_expand[:, 14:18])
                kmpip_mass = self.count_mass(
                    x_expand[:, 28:32] + x_expand[:, 10:14])
                x = x[torch.where((torch.abs(kppim_mass - 0.9) > 0.05)
                                  & (torch.abs(kmpip_mass - 0.9) > 0.05))]
                x_expand = x_expand[torch.where(
                    (torch.abs(kppim_mass - 0.9) > 0.05) & (torch.abs(kmpip_mass - 0.9) > 0.05))]
            free_targets = free_targets[:x_expand.shape[0]]
            targets = targets[:x_expand.shape[0]]
            loss = criterion(x,
                             x_expand,
                             free_targets,
                             targets,
                             return_detail=return_detail)
            if return_x:
                return loss, x_expand
            else:
                return loss
        else:
            if expand_function is None:
                return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            else:
                x_free = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                x_expand = expand_function(x_free)
                x_original = expand_function(x_free,
                                             return_four_momentum=True)
                if self.cut_theta_index is not None:
                    for cut_index in self.cut_theta_index:
                        cut_index1 = torch.where(
                            x_expand[:, cut_index] > 0.1 * torch.pi)
                        x_free = x_free[cut_index1]
                        x_expand = x_expand[cut_index1]
                        x_original = x_original[cut_index1]
                        cut_index2 = torch.where(
                            x_expand[:, cut_index] < 0.9 * torch.pi)
                        x_free = x_free[cut_index2]
                        x_expand = x_expand[cut_index2]
                        x_original = x_original[cut_index2]
                    kppim_mass = self.count_mass(
                        x_expand[:, 24:28] + x_expand[:, 14:18])
                    kmpip_mass = self.count_mass(
                        x_expand[:, 28:32] + x_expand[:, 10:14])
                    mass_cut_index = torch.where(
                        (torch.abs(kppim_mass - 0.9) > 0.05) & (torch.abs(kmpip_mass - 0.9) > 0.05))
                    x_free = x_free[mass_cut_index]
                    x_expand = x_expand[mass_cut_index]
                    x_original = x_original[mass_cut_index]
                if return_all:
                    return x_expand, x_free, x_original
                else:
                    return x_expand


if __name__ == '__main__':
    latent_dim = 100
    quantities_num = 8
    embedding_num = 1024
    nhead = 16
    dff = 4096
    norm = 'ln'
    activation = 'gelu'
    num_layers = 6
    x = torch.randn(1, latent_dim)
    model_obj_list = [
        TransformerGenerator(latent_dim, quantities_num, embedding_num, nhead,
                             dff, norm, activation, num_layers)
    ]
    for model_obj in model_obj_list:
        print(model_obj(x).shape)
