from os.path import join

import numpy as np
import pandas as pd

from torch.utils import data
from torchvision.datasets.folder import default_loader

from common import get_random_train_val_split_indices


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
        # Define paths
        root = join(root, 'herbarium_fertility')
        annotations_path = join(root, 'annotations')
        annotations_filename = join(annotations_path, 'metadata.csv')
        images_filename = join(annotations_path, 'image_filenames.csv')

        # Check if paths exist
        for path in [root, annotations_path, annotations_filename, images_filename]:
            if not os.path.exists(path):
                raise ValueError('could not found path: {}'.format(path))

        # Load CSV files
        df = pd.read_csv(annotations_filename, index_col='id')
        df_filenames = pd.read_csv(images_filename, index_col='id')
        df = df.merge(df_filenames, how='right', on='id',
                      validate='one_to_one')

        # Remove rows with missing values,
        # i.e. images that could not be fetched via their URL
        df.dropna(inplace=True)

        if task == 'fertility':
            filename = join(annotations_path, 'fertility_task.csv')
            df_task = pd.read_csv(filename, index_col='id')
            df = df.merge(df_task, how='right', on='id', validate='one_to_one')
            targets = df['is_fertile'].values.astype(np.float32).reshape(-1, 1)
            n_classes = 1
        elif task == 'flower/fruit':
            filename = join(annotations_path, 'flower_fruit_task.csv')
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

        self.image_files = join(root, 'images') + '/' + df['image_filename']
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
