import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from .kinematics import ExpandFourMomentumQuantities, GetFreeQuantities
from prefetch_generator import BackgroundGenerator


class IterableDataset(Dataset):

    def __init__(self, data, free_data, batch_event_num, len_dataset, latent_dim):
        self.data = data
        self.free_data = free_data
        self.batch_event_num = batch_event_num
        self.len_dataset = len_dataset
        self.len_data = free_data.shape[0]
        self.latent_dim = latent_dim

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        inputs_data = torch.rand(self.latent_dim) * 2 - 1
        index = torch.randint(low=0, high=self.len_data, size=(self.batch_event_num,))
        free_targets_data = self.free_data[index]
        targets_data = self.data[index]
        return inputs_data, free_targets_data, targets_data


def get_train_dataloader(path: str, epoch_iter_num: int, batch_event_num: int,
                         latent_dim: int, batch_size: int, num_workers: int,
                         pin_memory: bool, decay_num: int, dtype: str,
                         train_size: int, distribution_transform: bool):
    data = np.load(path).astype(dtype)[:train_size]
    train_data = torch.from_numpy(
        ExpandFourMomentumQuantities(decay_num=decay_num, distribution_transform=distribution_transform)(data))
    train_free_data = torch.from_numpy(GetFreeQuantities(data, distribution_transform=distribution_transform)())
    train_dataset = IterableDataset(data=train_data,
                                    free_data=train_free_data,
                                    batch_event_num=batch_event_num,
                                    len_dataset=epoch_iter_num * batch_size,
                                    latent_dim=latent_dim)
    train_dataloader = DataLoaderX(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory,
                                   persistent_workers=True)
    return train_dataloader


def get_validation_data(path: str, decay_num: int, dtype: str, distribution_transform: bool):
    data = np.load(path)
    return torch.from_numpy(
        ExpandFourMomentumQuantities(decay_num, distribution_transform=distribution_transform)
        (data).astype(dtype)), torch.from_numpy(
            GetFreeQuantities(data.astype(dtype), distribution_transform=distribution_transform)()), torch.from_numpy(data.astype(dtype))


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

