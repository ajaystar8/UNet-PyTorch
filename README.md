# UNet Paper Implementation using PyTorch

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](
https://github.com/ajaystar8/UNet-PyTorch/blob/main/LICENSE)


## Table of Contents

+ [About](#about)
  + [Dataset](#dataset)
+ [Getting Started](#getting_started)
  + [Project Structure](#dir_str)
  + [Prerequisites](#prereq)
+ [Installing the Requirements](#installing)
+ [Running the Code](#run_locally)
+ [TODO](#todo)
+ [A Kind Request](#request)
+ [License](#license)

## About <a name="about"></a>

A personal project to learn about semantic segmentation using PyTorch.

Implementation of the UNet architecture as described in [Ronneberger et al.](https://arxiv.org/abs/1505.04597) for task of semantic segmentation of the Humerus Bone using X-Ray images as the input modality. 

Reproduction of results is carried out using a subset of the [MURA](https://stanfordmlgroup.github.io/competitions/mura/) dataset.

The main code is located in the [train.py](train.py) file. All other code files are imported into [train.py](train.py) for training and testing the model.

The code to perform segmentation on custom images is present in [predict.py](predict.py) file. 

For your reference, the UNet architecture diagram (from [Ronneberger et al.](https://arxiv.org/abs/1505.04597)) is attached below.

![UNet Architecture Diagram](assets/architecture.png)

### Dataset <a name="dataset"></a> 

The images obtained from [MURA](https://stanfordmlgroup.github.io/competitions/mura/) had the X-Ray images included, without the ground truth segmentation masks. 

Hence, ground truth annotations were created using the [LabelMe](https://github.com/labelmeai/labelme.git) software. The created masks were later validated by medical professionals. 

## Getting Started <a name="getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine.

### Project Structure <a name="dir_str"></a>
The project is structured as follows:
```
UNet-PyTorch/
├── config/
│   └── __init__.py
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── models/
├── requirements.txt
├── utils.py
├── data_setup.py
├── model_builder.py
├── engine.py
├── train.py
└── predict.py     
```
Ensure that your directory structure abides by the structure mentioned above. Especially, make sure your data folder is structured in the format mentioned above. For your reference, an empty `data` directory following this structure is placed in this project.


### Prerequisites <a name = "prereq"></a>

You need to have a machine with Python > 3.6 and any Bash based shell (e.g. zsh) installed:

```ShellSession
$ python3.8 -V
Python 3.8.18

$ echo $SHELL
/bin/zsh
```

## Installing the Requirements <a name="installing"></a>

Clone the repository: 
```ShellSession
$ git clone https://github.com/ajaystar8/UNet-PyTorch.git
```

Install requirements using in a new conda environment:
```ShellSession
$ conda create -n name_of_env python=3.8 --file requirements.txt
```

## Running the Code <a name="run_locally"></a>

Navigate to the [config](config/__init__.py) package and specify the following: 
+ Path to your `data` directory.
+ Path to `models` directory for saving model checkpoints.
+ Change other `hyperparameters` if necessary.

Activate the conda environment:
```ShellSession
$ conda activate name_of_env
```

To start training the model, you can call the [train.py](train.py) script. Efforts have been taken to ensure that most of the parameters and hyperparameters to train and test the model can be set manually. You can get the list of command line arguments that can be toggled by executing the command:

```ShellSession
$ python3 train.py --help

usage: train.py [-h] [-v VERBOSITY] [--input_dims H W] [--epochs NUM_EPOCHS] [--batch_size N] [--learning_rate LR] [--in_channels IN_C] [--out_channels OUT_C]
                DATA_DIR CHECKPOINT_DIR RUN_NAME DATASET_NAME WANDB_API_KEY

Script to begin training and validation of UNet.

positional arguments:
  DATA_DIR              path to dataset directory
  CHECKPOINT_DIR        path to directory storing model checkpoints
  RUN_NAME              Name of current run
  DATASET_NAME          Name of dataset over which model is to be trained
  WANDB_API_KEY         API key of your Weights and Biases Account.

optional arguments:
  -h, --help            show this help message and exit
  -v VERBOSITY, --verbose VERBOSITY
                        setting verbosity to 1 will send email alerts to user after every epoch (default: 0)

Hyperparameters for model training:
  --input_dims H W      spatial dimensions of input image (default: [256, 256])
  --epochs NUM_EPOCHS   number of epochs to train (default: 10)
  --batch_size N        number of images per batch (default: 1)
  --learning_rate LR    learning rate for training (default: 0.0001)

Architecture parameters:
  --in_channels IN_C    number of channels in input image (default: 1)
  --out_channels OUT_C  number of classes in ground truth mask (default: 1)

Happy training! :)
```

The command shown below is an example of a call that can be used to train the model.
```ShellSession
$ python3 train.py --verbose 0 --input_dims 256 256 --epochs 30 --batch_size 2 --loss_fn BCEWithLogitsLoss --learning_rate 1e-5 --exp_track false --in_channels 1 --out_channels 1 ./data ./models sample_run MURA WANDB_KEY 
```

## TODO <a name="todo"></a>

Read the [TODO](TODO.md) to see the current task list. 

## A Kind Request <a name="request"></a>

I have tried to adopt good coding practices as mentioned in different blogs and articles. 
However, I feel there is still a lot of room for improvement in making the code more efficient, 
modular and easy to understand.

I would be thankful if you could share your opinions by opening a GitHub Issue for the same. Your
criticisms are always welcome! 


## License <a name = "license"></a>

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

