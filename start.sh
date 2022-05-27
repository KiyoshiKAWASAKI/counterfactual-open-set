#!/bin/bash
# Break on any error
set -e


# Hyperparameters
GAN_EPOCHS=3
CLASSIFIER_EPOCHS=3
CF_COUNT=100
GENERATOR_MODE=open_set
DATASET_DIR=/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24
RESULT_DIR=/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/osrci/debug

# 1. Train the initial generative model (E+G+D) and the initial classifier (C_K)
python generativeopenset/train_gan.py --epochs $GAN_EPOCHS --result_dir $RESULT_DIR

# 2. Baseline: Evaluate the standard classifier (C_k+1)
#python generativeopenset/evaluate_classifier.py --result_dir . --mode baseline
cp checkpoints/classifier_k_epoch_000${GAN_EPOCHS}.pth checkpoints/classifier_kplusone_epoch_000${GAN_EPOCHS}.pth

# 3. Generate a number of counterfactual images (in the K+2 by K+2 square grid format)
#python generativeopenset/generate_${GENERATOR_MODE}.py --result_dir . --count $CF_COUNT

# 4. Automatically label the rightmost column in each grid (ignore the others)
#python generativeopenset/auto_label.py --output_filename generated_images_${GENERATOR_MODE}.dataset

# 5. Train a new classifier, now using the aux_dataset containing the counterfactuals
#python generativeopenset/train_classifier.py --epochs $CLASSIFIER_EPOCHS --aux_dataset generated_images_${GENERATOR_MODE}.dataset

# 6. Evaluate the C_K+1 classifier, trained with the augmented data
#python generativeopenset/evaluate_classifier.py --result_dir . --mode fuxin

#./print_results.sh
