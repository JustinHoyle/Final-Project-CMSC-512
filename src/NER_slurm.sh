#!/bin/bash
#SBATCH --job-name=hf_ner_job
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

echo "Starting NER..."

python named_entity_recognition.py --input SentimentTwitterDataset.csv --output pii_detected_tweets_unclean.csv

echo "NER complete."
