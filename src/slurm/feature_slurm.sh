#!/bin/bash
#SBATCH --job-name=hf_feature_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=hf_ner_%j.out
#SBATCH --error=hf_ner_%j.err
#SBATCH --mail-user=hoylejb2@vcu.edu
#SBATCH --mail-type=END,FAIL

# Activate virtual environment if necessary
source /lustre/home/hoylejb2/final-project/.venv/bin/activate

echo "Starting feature processing..."

python feature_processing.py --input src/pii_detected_tweets.csv

echo "Processing complete."
