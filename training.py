import os
import sys
from os.path import join

import numpy as np
import torch
from torch import nn, optim
from torchvision import models

from common import get_data_transformation
from datasets import herbarium_fertility, herbarium_phenophase
from helpers import binary_accuracy, multiclass_accuracy, train
from utils import print_cuda_info, get_default_data_loaders


def train_command(args):
    if not os.path.exists(args.experiment_output_path):
        os.makedirs(args.experiment_output_path)

    # Define a logger
    class Logger(object):
        def __init__(self, log_path):
            self.terminal = sys.stdout
            self.log = open(join(log_path, 'train.log'), 'w')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.terminal.flush()
            self.log.flush()

        def flush(self):
            # Needed for python 3 compatibility.
            pass

    sys.stdout = Logger(args.experiment_output_path)

    print(' '.join(sys.argv), '\n')

    print_cuda_info()

    # Preprocessing and data data augmentation
    train_transform, test_transform = \
        get_data_transformation(args.keep_image_ratio, args.downsample_image)

    # Load dataset
    print('\n# Loading dataset')
    if args.task == 'phenophase':
        dataset = herbarium_phenophase
        (train_data_loader, test_data_loader), (n_samples_train, n_samples_test) =\
            get_default_data_loaders(
                dataset,
                batch_size=args.batch_size,
                train_transform=train_transform,
                test_transform=test_transform,
                test=False,
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
                test=False,
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

    # Load pretrained model
    print('\n# Loading model')
    model = getattr(models, args.model)(pretrained=True)
    if args.model == 'inception_v3':
        raise ValueError(
            'InceptionV3 not supported due to too many differences with other '
            'models (i.e. input size of 299x299, auxiliary classifiers, etc.)')

    # Adapt last average pooling layer to different image sizes
    if args.keep_image_ratio:
        if args.downsample_image:
            model.avgpool = nn.AvgPool2d(kernel_size=(13, 8), stride=1)
        else:
            model.avgpool = nn.AvgPool2d(kernel_size=(27, 18), stride=1)

    n_classes = train_data_loader.dataset.n_classes

    n_outputs = n_classes
    model.fc = nn.Linear(model.fc.in_features, n_outputs)
    clf = model.fc
    nn.init.kaiming_normal_(clf.weight)
    nn.init.constant_(clf.bias, val=0)
    print(model)

    if args.task == 'phenophase':
        criterion = nn.CrossEntropyLoss()
        metric = multiclass_accuracy
    else:
        criterion = nn.BCEWithLogitsLoss()
        metric = binary_accuracy
    print(criterion)

    params = model.parameters()

    print('\n# Finetuning whole network...')
    optimizer = optim.SGD(params, lr=args.lr,
                          momentum=.9, nesterov=True)
    print(optimizer)

    if args.lr_decay:
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = np.asarray(eval(args.lr_decay))
        if issubclass(milestones.dtype.type, np.floating):
            milestones = (args.num_epochs * milestones).astype(np.int)
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        print '{}(milestones={}, gamma={})'.format(
            lr_scheduler.__class__.__name__, lr_scheduler.milestones,
            lr_scheduler.gamma
        )
    else:
        lr_scheduler = None

    history = train(
        model, optimizer, criterion, train_data_loader,
        n_epochs=args.num_epochs, lr_scheduler=lr_scheduler,
        metrics=[metric], val_data_loader=test_data_loader, gpu=True
    )

    with open(join(args.experiment_output_path, 'config.txt'), 'w') as f:
        f.write(repr(train_data_loader.dataset) + '\n')
        f.write(repr(model) + '\n')
        f.write(repr(criterion) + '\n')
        f.write(repr(optimizer) + '\n')

    import pandas as pd
    df = pd.DataFrame(history)
    df.to_csv(join(args.experiment_output_path, 'training.csv'),
              index_label='epoch')

    torch.save(model, join(args.experiment_output_path, 'model.pkl'))
