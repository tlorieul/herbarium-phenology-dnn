import argparse

from training import train_command
from prediction import predict_command


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dataset_root', type=str, required=True, default=argparse.SUPPRESS,
        help='path to dataset'
    )
    parser.add_argument(
        '--task', type=str, choices=["fertility", "flower/fruit"],
        help='which task to use for biggest dataset'
    )
    parser.add_argument(
        '--subset', type=str,
        choices=["train", "test", "random_test", "species_test", "herbarium_test"],
        required=True, default=argparse.SUPPRESS,
        help='which subset to use'
    )
    parser.add_argument(
        '--batch_size', type=int, required=True, default=argparse.SUPPRESS,
        help='training batch size'
    )
    parser.add_argument(
        '--keep_image_ratio', action='store_true',
        help='image preprocessing that preserves the image ratio'
    )
    parser.add_argument(
        '--downsample_image', action='store_true',
        help='image preprocessing that preserves downsamples the image by a '
             'factor of 2'
    )
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='number of jobs for data loading'
    )

    subparsers = parser.add_subparsers(help='action: train or predict')

    # Subparser for training
    parser_train = subparsers.add_parser(
        'train', help='perform training'
    )
    parser_train.add_argument('experiment_output_path')
    parser_train.add_argument(
        '--model', type=str, default='resnet50',
        help='model to finetune'
    )
    parser_train.add_argument(
        '--num_epochs', type=int, required=True, default=argparse.SUPPRESS,
        help='max number of epochs for training'
    )
    parser_train.add_argument(
        '--lr', type=float, required=True, default=argparse.SUPPRESS,
        help='learning rate'
    )
    parser_train.add_argument(
        '--data_augmentation', action='store_true',
        help='data augmentation to use during training'
    )
    parser_train.set_defaults(func=train_command)

    # Subparser for prediction
    parser_predict = subparsers.add_parser(
        'predict', help='predict on val/test'
    )
    parser_predict.add_argument('model_file')
    parser_predict.add_argument('output_predictions_file')
    parser_predict.set_defaults(func=predict_command)

    args = parser.parse_args()

    # Delegate to action handler
    args.func(args)
