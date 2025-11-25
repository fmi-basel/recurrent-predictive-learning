# Recurrent Predictive Learning (RPL)
This repository contains code for reproducing results reported in our preprint on RPL,
a JEPA model of predictive processing.
## Setup
We have provided the code as a python package. To setup your python environment 
and run the code, clone this repository and follow the following steps:

    # Go to the directory
    cd recurrent-predictive-learning
    # Install the package including all the dependencies via pip
    pip install .

We recommend to install the package in a separate project-specific virtual environment.
Note that you will need a CUDA-compatible hardware to run the code smoothly and
`mps` support for Mac users is currently unavailable.
## Usage
You can train all models for the corresponding data using the provided bash scripts:
- `bash bash_scripts/run_moving_animals.sh` trains and evaluates all models on the moving animal videos
 (cf. Fig. 2 and Supplementary Fig. S2)
- `bash bash_scripts/run_mnist_triplets.sh` trains and evaluates all models on the MNIST triplet sequences
 (cf. Fig. 4)
- `bash bash_scripts/run_mouse.sh` trains and evaluates all models on the videos of behaving mice
 (cf. Fig. 5a,b)
- `bash bash_scripts/run_libri.sh` trains all models on the Librispeech corpus (cf. Fig 5c,d)
- `bash bash_scripts/run_oddballs.sh` trains the model for the local-global oddball paradigm
 (cf. Fig. 7)
- `bash bash_scripts/run_hRPL.sh` trains and evaluates hRPL models and the frozen control models
 on the moving animal videos (cf. Fig. 8)

To evaluate each model individually, you can use:

    # For RPL
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 repl/scripts/offline_eval.py [options]
    # For hRPL
    torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 repl/scripts/offline_greedy_eval.py [options]

Use `--help` to see all available arguments and make sure to pass `--model_path <PATH-TO-MODEL>/model_final.pt`.
## Reproduction of figures
Use the provided jupyter notebooks in the `notebooks` folder to generate the figures
in the paper or do further analysis. The notebooks are named by referring to the
corresponding figure number in the manuscript.
