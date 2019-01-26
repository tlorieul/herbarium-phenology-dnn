# Requirements

This has been tested using:

* Python 2.7
* Pytorch 0.4.0
* CUDA 9.1

Other dependencies are:

* numpy
* torchvision

# Usage

See help for usage instructions:

```bash
python herbarium_phenology_dnn.py -h
usage: herbarium_phenology_dnn.py [-h] --dataset_root DATASET_ROOT
                                  [--task {fertility,flower/fruit}] --subset
                                  {train,test,random_test,species_test,herbarium_test}
                                  --batch_size BATCH_SIZE [--keep_image_ratio]
                                  [--downsample_image]
                                  [--num_workers NUM_WORKERS]
                                  {train,predict} ...

positional arguments:
  {train,predict}       action: train or predict
    train               perform training
    predict             predict on val/test

optional arguments:
  -h, --help            show this help message and exit
  --dataset_root DATASET_ROOT
                        path to dataset
  --task {fertility,flower/fruit}
                        which task to use for biggest dataset (default: None)
  --subset {train,test,random_test,species_test,herbarium_test}
                        which subset to use
  --batch_size BATCH_SIZE
                        training batch size
  --keep_image_ratio    image preprocessing that preserves the image ratio
                        (default: False)
  --downsample_image    image preprocessing that preserves downsamples the
                        image by a factor of 2 (default: False)
  --num_workers NUM_WORKERS
                        number of jobs for data loading (default: 8)
```

For help about how to perform training:

```bash
python herbarium_phenology_dnn.py train -h
usage: herbarium_phenology_dnn.py train [-h] [--model MODEL] --num_epochs
                                        NUM_EPOCHS --lr LR
                                        [--data_augmentation]
                                        experiment_output_path

positional arguments:
  experiment_output_path

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to finetune
  --num_epochs NUM_EPOCHS
                        max number of epochs for training
  --lr LR               learning rate
  --data_augmentation   data augmentation to use during training
```

For help about how to perform prediction:

```bash
python herbarium_phenology_dnn.py predict -h
usage: herbarium_phenology_dnn.py predict [-h]
                                          model_file output_predictions_file

positional arguments:
  model_file
  output_predictions_file

optional arguments:
  -h, --help            show this help message and exit
```

# Reproductibility instruction

Here are the commands that have to be executed in order to reproduce the results from the paper.

## Download and format datasets

The training was performed on "Large-scale and fine-grained phenological stage annotation of herbarium specimens datasets".

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2548630.svg)](https://doi.org/10.5281/zenodo.2548630)

TODO

## Training phase

For EXP1-Fertility ResNet50-Large:

```bash
python herbarium_phenology_dnn.py --dataset_root <path_to_dataset> --task fertility --subset train --batch_size 48 --keep_image_ratio --downsample_image train --model resnet50 --num_epochs 45 --lr 0.001 --data_augmentation
```

For EXP1-Fertility ResNet50-VeryLarge:

```bash
python herbarium_phenology_dnn.py --dataset_root <path_to_dataset> --task fertility --subset train --batch_size 12 --keep_image_ratio train --model resnet50 --num_epochs 45 --lr 0.001 --data_augmentation
```

For EXP2-Fl.Fr ResNet50-VeryLarge:

```bash
python herbarium_phenology_dnn.py --dataset_root <path_to_dataset> --task "flower/fruit" --subset train --batch_size 12 --keep_image_ratio train --model resnet50 --num_epochs 45 --lr 0.01 --data_augmentation
```

For EXP3-Pheno ResNet50-VeryLarge:

```bash
python herbarium_phenology_dnn.py --dataset_root <path_to_dataset> --subset train --batch_size 8 --keep_image_ratio train --model resnet50 --num_epochs 30 --lr 0.001 --data_augmentation
```

## Test phase

In order to perform the predictions on the different subsets and to save them on the disk, the following command should be executed.

```bash
python herbarium_phenology_dnn.py --dataset_root <path_to_dataset> --subset <subset> --batch_size 128 --keep_image_ratio predict <model_file> <output_predictions_file>
```

# Using learned models for transfer learning

The models provided in the `models` folder can be loaded using
```python
import torch
torch.load(model_filename)
```

They can then be finetuned on other datasets, using the learned parameters as initialization of the new model.

# TODO

* [ ] add script to automatically download datasets and format them properly
* [ ] make containers for easier distribution and reproductibility
* [ ] improve documentation
* [ ] add compatibility with Python 3.6+ and Pytorch 1.0+

