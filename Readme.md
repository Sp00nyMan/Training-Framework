The repository contains two branches:
 - master - includes the GLOW variants
 - rnvp - includes the RealNVP variants


The training of the models can be performed using the train.py script.

For instructions on how to run the script, run python train.py --help.

## Requirements

python==3.12.3
torch==2.2.2
torchvision
torch-fidelity
tqdm
wandb
timm
