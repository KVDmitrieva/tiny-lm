#!/bin/bash

echo "Download data"
mkdir data
wget "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt?download=true" -O data/train_raw.txt
wget "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt?download=true" -O data/val_raw.txt

echo "Process raw files"
python preprocessing.py

echo "Remove raw files"
rm data/train_raw.txt
rm data/val_raw.txt

echo "Done!"