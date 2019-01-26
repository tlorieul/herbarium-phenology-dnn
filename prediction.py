import numpy as np

import torch

from common import get_data_transformation
from datasets import herbarium_fertility, herbarium_phenophase
from helpers import predict
from utils import print_cuda_info, get_default_data_loaders


def predict_command(args):
    print_cuda_info()

    # Preprocessing and data data augmentation
    train_transform, test_transform = \
        get_data_transformation(args.keep_image_ratio, args.downsample_image)

    # Load dataset
    if args.task == 'phenophase':
        dataset = herbarium_phenophase
        (train_data_loader, test_data_loader), (n_samples_train, n_samples_test) =\
            get_default_data_loaders(
                dataset,
                batch_size=args.batch_size,
                train_transform=train_transform,
                test_transform=test_transform,
                test=True,
                num_workers=args.num_workers,
                root=args.dataset_root,
                subset=args.subset
        )
    else:
        dataset = herbarium_fertility
        (train_data_loader, test_data_loader), (n_samples_train, n_samples_test) =\
            get_default_data_loaders(
                dataset,
                batch_size=args.batch_size,
                train_transform=train_transform,
                test_transform=test_transform,
                test=True,
                num_workers=args.num_workers,
                root=args.dataset_root,
                task=args.task,
                subset=args.subset
        )
    print('Train dataset: {}'.format(train_data_loader.dataset))
    print('Train sampler: {}'.format(train_data_loader.sampler.__class__.__name__))
    print('Train set size: {}'.format(n_samples_train))
    print('Test dataset: {}'.format(test_data_loader.dataset))
    print('Test sampler: {}'.format(test_data_loader.sampler.__class__.__name__))
    print('Test set size: {}'.format(n_samples_test))

    model = torch.load(args.model_file)
    preds = predict(model, test_data_loader, gpu=True)
    np.save(args.output_predictions_file, preds)
