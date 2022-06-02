#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044
#$ -e errors/
#$ -N osrci_seed_4

# Required modules
module load conda
conda init bash
source activate new_msd_net

python ./generativeopenset/pipeline.py