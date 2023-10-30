import torch
from torch import Tensor
from torch.nn import Module
from numpy import array
import numpy as np
from tqdm import tqdm

epsilon = 1e-12


class SWDLoss(Module):

    def __init__(
        self,
        gamma: int,
        weight_gamma: int = None,
        weight_index: list = None,
        repeat_projector_num: int = 1,
        projector_dim: int = 512
    ):
        super().__init__()
        self.gamma = gamma
        self.weight_gamma = weight_gamma
        if weight_index is None:
            self.weight_index = None
        else:
            self.weight_index = torch.tensor(weight_index)
        self.repeat_projector_num = repeat_projector_num
        self.projector_dim = projector_dim

    def projector(self, x, y, random_projector=None):
        if random_projector is None:
            random_projector = torch.randn(x.shape[1],
                                           self.projector_dim, device=x.device)
            random_projector /= torch.sqrt(
                torch.sum(random_projector ** 2, dim=0, keepdim=True))
        else:
            assert random_projector.shape[0] == x.shape[1]
            random_projector = random_projector.to(x.device)
        x = torch.matmul(x, random_projector)
        y = torch.matmul(y, random_projector)
        return x, y

    def wd(self, x_y, p=1, dim=0, require_sort=True, return_list=False):
        x, y = x_y
        if require_sort:
            x = torch.sort(x, dim=0)[0]
            y = torch.sort(y, dim=0)[0]
        if p == 1:
            loss = torch.mean(torch.abs(x - y), dim=dim)
        elif p == 2:
            loss = torch.mean(torch.square(x - y), dim=dim)
        if return_list:
            return loss
        else:
            return loss.mean()

    def random_permutation(self, z, sort_index, x_index):
        mask_x = torch.zeros_like(z, dtype=torch.bool)
        mask_x[:, x_index] = 1
        mask_x_sorted = torch.gather(mask_x, 1, sort_index)
        x_sorted = z[mask_x_sorted]
        y_sorted = z[~mask_x_sorted]
        x_sorted = x_sorted.reshape(z.shape[0], x_index.shape[0])
        y_sorted = y_sorted.reshape(z.shape[0], z.shape[1] - x_index.shape[0])
        return x_sorted, y_sorted

    def count_multi_dimensional_p_value(self, x, y, p, x_index_list):
        x = x.transpose(1, 0)
        y = y.transpose(1, 0)
        z = torch.cat([x, y], dim=1)
        z, sort_index = torch.sort(z, dim=1)
        x, y = self.random_permutation(z, sort_index, torch.arange(x.shape[1]))
        original_distance = self.wd((x, y), p=p, dim=1, require_sort=False, return_list=False).item()
        distance_list = np.empty(x_index_list.shape[0])
        for i, x_index in tqdm(enumerate(x_index_list), total=x_index_list.shape[0], desc='count p value'):
            x, y = self.random_permutation(z, sort_index, x_index)
            distance_list[i] = self.wd((x, y), p=p, dim=1, require_sort=False, return_list=False)
        return original_distance, np.mean(distance_list > original_distance)

    def count_p_value(self, free_predicts: Tensor,
                      predicts: Tensor,
                      free_targets: Tensor,
                      targets: Tensor,
                      permutation_list: list,
                      random_projection,
                      return_original_loss=True):
        random_projection_predicts = (free_predicts @ random_projection).half()
        predicts = predicts.half()
        random_projection_targets = (free_targets @ random_projection).half()
        targets = targets.half()
        original_swd_loss, swd_p_value = self.count_multi_dimensional_p_value(random_projection_predicts, random_projection_targets, p=1, x_index_list=permutation_list[:, :predicts.shape[0]])
        original_wd_loss, wd_p_value = self.count_multi_dimensional_p_value(predicts, targets, p=1, x_index_list=permutation_list[:, :predicts.shape[0]])
        if return_original_loss:
            return original_wd_loss, original_swd_loss, wd_p_value, swd_p_value
        else:
            return wd_p_value, swd_p_value

    def count_one_dimensional_p_value(self, predicts: array, targets: array, permutation_list):
        predicts = torch.from_numpy(predicts).half().cuda()
        targets = torch.from_numpy(targets).half().cuda()
        z = torch.cat([predicts, targets], dim=0)
        z, z_index = torch.sort(z, dim=0)
        predicts_size = predicts.shape[0]
        original_distance = self.wd(
            (z[z_index < predicts_size], z[z_index >= predicts_size]), p=1, require_sort=False).item()
        predicts = []
        targets = []
        distance_list = np.empty(permutation_list.shape[0])
        for i, z_index in enumerate(permutation_list):
            distance_list[i] = self.wd((z[z_index < predicts_size], z[z_index >= predicts_size]), p=1, require_sort=False, return_list=True).detach().cpu().numpy()
        p_value = np.mean(distance_list >= original_distance)
        return p_value

    def forward(self,
                free_predicts: Tensor,
                predicts: Tensor,
                free_targets: Tensor,
                targets: Tensor,
                random_projection=None,
                return_detail: bool = False
                ):
        """
        shape: (numbers of events, numbers of quantities)
        """
        assert predicts.shape == targets.shape
        assert free_predicts.shape == free_targets.shape
        if self.gamma == 0:
            wd_loss = torch.tensor([0.], device=predicts.device)
        else:
            if self.weight_index is not None and self.weight_gamma != 0:
                wd_loss = self.wd(
                    (targets, predicts), p=1) + self.weight_gamma * self.wd(
                    (targets[:, self.weight_index],
                     predicts[:, self.weight_index]), p=1)
            else:
                wd_loss = self.wd(
                    (targets, predicts), p=1)
        swd_loss = 0
        if random_projection is not None:
            for i in range(random_projection.shape[0]):
                swd_loss += self.wd(
                    self.projector(torch.arctanh((1 - epsilon) * free_targets), torch.arctanh((1 - epsilon) * free_predicts),
                                   random_projection[i]), p=1)
            swd_loss /= random_projection.shape[0]
        else:
            for _ in range(self.repeat_projector_num):
                swd_loss += self.wd(
                    self.projector(torch.arctanh((1 - epsilon) * free_targets), torch.arctanh((1 - epsilon) * free_predicts)), p=1)
            swd_loss /= self.repeat_projector_num
        total_loss = self.gamma * wd_loss + swd_loss
        if return_detail:
            return total_loss, wd_loss, swd_loss
        else:
            return total_loss
