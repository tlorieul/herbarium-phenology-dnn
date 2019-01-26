from os.path import join

import numpy as np
import pandas as pd

from torch.utils import data
from torchvision.datasets.folder import default_loader


import sys
sys.path.append('../dirichlet_loss')

from toolkit.common.torch.utils import get_random_train_val_split_indices


def get_transform(data_augmentation=False):
    raise NotImplementedError()


def get_dataset(**kwargs):
    return HerbariumDataset(**kwargs)


def get_iid_train_val_split(train_dataset):
    return get_random_train_val_split_indices(
        train_dataset, test_size=.1, shuffle=True, random_seed=0)


def get_species_train_val_split():
    raise NotImplementedError()


def get_herbarium_train_val_split():
    raise NotImplementedError()


get_train_val_split_indices = get_iid_train_val_split


class HerbariumDataset(data.Dataset):
    def __init__(self, root, task, train, subset,
                 transform=None, target_transform=None):

        filename = join(root, 'metadata.csv')
        df = pd.read_csv(filename, index_col='id')

        if task == 'fertility':
            filename = join(root, 'fertility_task.csv')
            df_task = pd.read_csv(filename, index_col='id')
            df = df.merge(df_task, how='right', on='id', validate='one_to_one')
            targets = df['is_fertile'].values.astype(np.float32).reshape(-1, 1)
            n_classes = 1
        elif task == 'flower/fruit':
            filename = join(root, 'flower_fruit_task.csv')
            df_task = pd.read_csv(filename, index_col='id')
            df = df.merge(df_task, how='right', on='id', validate='one_to_one')
            targets = df[['has_flower', 'has_fruit']].values.astype(np.float32)
            n_classes = 2
        else:
            raise ValueError(
                'task must be equale to "fertility" or "flower/fruit", '
                'given {}'.format(task))

        available_subsets = ['train', 'random_test', 'species_test',
                             'herbarium_test']
        if subset not in available_subsets:
            raise ValueError(
                'subset must be one of: {}'.format(available_subsets))

        set_ind = df['train_test_set'] == subset
        df = df[set_ind]
        targets = targets[set_ind]

        self.image_files = root + '/images/' + df['collection'] + '/' + df['image_filename']
        self.image_files = self.image_files.values

        self.root = root
        self.train = train
        self.df = df
        self.targets = targets
        self.n_classes = n_classes

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.image_files[index]
        target = self.targets[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.image_files)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
