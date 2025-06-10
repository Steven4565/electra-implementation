#!/bin/bash

echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt || exit 1
pip install gdown || exit 1

echo "Downloading datasets..."
python pretraining/download_datasets.py --glue || exit 1

echo "Downloading model files"
gdown --folder --id 1tASJhesjZEL9rbKuVqOXCVjcd8f2IryI

echo "Moving model files"
mkdir -p ./output/ckpt/final
mv ./PretrainedWeights/* ./output/ckpt/final
rm -r ./PretrainedWeights/

echo "Setup complete"
