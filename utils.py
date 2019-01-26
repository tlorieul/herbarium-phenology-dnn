import os

import torch
from torch.utils import data
from torch.utils.data.sampler import Sampler


def print_cuda_info():
    print('Using PyTorch version {}'.format(torch.__version__))
    print('CUDA available: {} (version: {})'.format(
        torch.cuda.is_available(), torch.version.cuda))
    print('cuDNN available: {} (version: {})'.format(
        torch.backends.cudnn.enabled, torch.backends.cudnn.version()))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('CUDA_VISIBLE_DEVICES: {}'.format(
            os.environ['CUDA_VISIBLE_DEVICES']))


class SubsetSequentialSampler(Sampler):
    """Samples elements from a given list of indices sequentially, always in the
    same order.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_default_data_loaders(dataset, batch_size, test, predict=False,
                             data_augmentation=False, num_workers=4,
                             train_transform=None, test_transform=None,
                             **kwargs):
    if train_transform is None:
        train_transform = dataset.get_transform(data_augmentation)
    if test_transform is None:
        test_transform = dataset.get_transform()

    train_dataset = dataset.get_dataset(train=True, transform=train_transform,
                                        **kwargs)
    test_dataset = dataset.get_dataset(train=not test,
                                       transform=test_transform, **kwargs)

    if test:
        if not predict:
            train_sampler = data.sampler.RandomSampler(train_dataset)
        else:
            train_sampler = data.sampler.SequentialSampler(train_dataset)
        test_sampler = data.sampler.SequentialSampler(test_dataset)
    else:
        train_ind, val_ind = dataset.get_train_val_split_indices(train_dataset)
        if not predict:
            train_sampler = data.sampler.SubsetRandomSampler(train_ind)
        else:
            train_sampler = SubsetSequentialSampler(train_ind)
        test_sampler = SubsetSequentialSampler(val_ind)

    sample_sizes = len(train_sampler), len(test_sampler)

    train_data_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True)
    test_data_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=True)

    return (train_data_loader, test_data_loader), sample_sizes
