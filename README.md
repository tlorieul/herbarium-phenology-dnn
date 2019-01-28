[![DOI](https://zenodo.org/badge/167395031.svg)](https://zenodo.org/badge/latestdoi/167395031)

# Requirements

This has been tested using:

* Python 2.7
* Pytorch 0.4.0
* CUDA 9.1
* cuDNN 7.0

Other dependencies are:

* Numpy
* Pandas
* Torchvision

For the download and formating of the dataset, the following additional dependencies might be required:

* Joblib for parallelization
* PIL for image integrity check
* ImageMagick for image preprocessing

# Usage

See help for usage instructions:

```
python herbarium_phenology_dnn.py -h
usage: herbarium_phenology_dnn.py [-h] --dataset_root DATASET_ROOT --task
                                  {fertility,flower/fruit,phenophase} --subset
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
                        path to datasets
  --task {fertility,flower/fruit,phenophase}
                        which task to use for biggest dataset
  --subset {train,test,random_test,species_test,herbarium_test}
                        which subset to use
  --batch_size BATCH_SIZE
                        training batch size
  --keep_image_ratio    image preprocessing that preserves the image ratio
                        (default: False)
  --downsample_image    image preprocessing that downsamples the
                        image by a factor of 2 (default: False)
  --num_workers NUM_WORKERS
                        number of jobs for data loading (default: 8)
```

For help about how to perform training:

```
python herbarium_phenology_dnn.py train -h
usage: herbarium_phenology_dnn.py train [-h] [--model MODEL] --num_epochs
                                        NUM_EPOCHS --lr LR
                                        [--data_augmentation]
                                        experiment_output_path

positional arguments:
  experiment_output_path

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to finetune (default: resnet50)
  --num_epochs NUM_EPOCHS
                        max number of epochs for training
  --lr LR               learning rate
  --data_augmentation   data augmentation to use during training (default:
                        False)
```

For help about how to perform prediction:

```
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

Note that as the images have to be downloaded from their URLs, some of them might not be accessible anymore.
This and the fact that there are some random fluctuations in the training of neural networks imply that executing the following commands might actually result in some slightly different values than those presented in the paper.

## Download and format datasets

The training was performed on "Large-scale and fine-grained phenological stage annotation of herbarium specimens datasets".

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2548630.svg)](https://doi.org/10.5281/zenodo.2548630)

Note that these steps may take a few hours and need free space to store the images.

For EXP1-Fertility and EXP2-Fl.Fr, this requires around 25G of free space:

```
python download_and_format_datasets.py --check_integrity --preprocess --n_jobs <number_of_jobs> herbarium_fertility <path_to_dataset>
```

For EXP3-Pheno, this requires around 2.6G of free space:

```
python download_and_format_datasets.py --check_integrity --preprocess --n_jobs <number_of_jobs> herbarium_asteraceae_phenophase <path_to_dataset>
```

This commandlines check if the images are already downloaded before downloading them.
It is thus possible to kill these commands and re-execute them in order to resume download.

These scripts display the number of images that could not be fetched properly.
It may be needed to rerun these scripts in order to fetch images that could not be downloaded properly the first time.

This script also has help information:

```
python download_and_format_datasets.py -h
usage: download_and_format_datasets.py [-h] [--check_integrity] [--preprocess]
                                       [--n_jobs N_JOBS]
                                       {herbarium_fertility,herbarium_asteraceae_phenophase}
                                       datasets_path

Downloads and formats herbarium phenology datasets

positional arguments:
  {herbarium_fertility,herbarium_asteraceae_phenophase}
                        name of the dataset to download
  datasets_path         where to save the datasets

optional arguments:
  -h, --help            show this help message and exit
  --check_integrity     check integrity of images after download (default:
                        False)
  --preprocess          preprocess the images by resizing them to 900x600 and
                        setting JPEG quality to 85 (default: False)
  --n_jobs N_JOBS       use several jobs to speed-up images download (default:
                        1)
```

## Training phase

For EXP1-Fertility ResNet50-Large:

```
python herbarium_phenology_dnn.py --dataset_root <path_to_dataset> --task fertility --subset train --batch_size 48 --keep_image_ratio --downsample_image train --model resnet50 --num_epochs 45 --lr 0.001 --data_augmentation <output_path>
```

For EXP1-Fertility ResNet50-VeryLarge:

```
python herbarium_phenology_dnn.py --dataset_root <path_to_dataset> --task fertility --subset train --batch_size 12 --keep_image_ratio train --model resnet50 --num_epochs 45 --lr 0.001 --data_augmentation <output_path>
```

For EXP2-Fl.Fr ResNet50-VeryLarge:

```
python herbarium_phenology_dnn.py --dataset_root <path_to_dataset> --task "flower/fruit" --subset train --batch_size 12 --keep_image_ratio train --model resnet50 --num_epochs 45 --lr 0.01 --data_augmentation <output_path>
```

For EXP3-Pheno ResNet50-VeryLarge:

```
python herbarium_phenology_dnn.py --dataset_root <path_to_dataset> --task phenophase --subset train --batch_size 8 --keep_image_ratio train --model resnet50 --num_epochs 30 --lr 0.001 --data_augmentation <output_path>
```

## Test phase

In order to perform the predictions on the different subsets and to save them on the disk, the following command should be executed.

```
python herbarium_phenology_dnn.py --dataset_root <path_to_dataset> --task <task> --subset <subset> --batch_size 128 --keep_image_ratio predict <model_file> <output_predictions_file>
```

# Using learned models for transfer learning

The models provided in the `models` folder can be loaded using
```python
import torch
torch.load(model_filename)
```

They can then be finetuned on other datasets, using the learned parameters as initialization of the new model.

# TODO

* [x] add script to automatically download datasets and format them properly
* [ ] make containers for easier distribution and reproductibility
* [ ] improve documentation
* [ ] add compatibility with Python 3.6+ and Pytorch 1.0+

