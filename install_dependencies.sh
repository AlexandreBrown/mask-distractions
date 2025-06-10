#!/bin/bash
pip install -r ./baselines/madi/requirements.txt

DATA_DIR=./datasets/

if [ -d "$DATA_DIR" ] && [ "$(ls -A "$DATA_DIR")" ]; then
  echo "Place365 dataset already present in $DATA_DIR; skipping download."
  exit 0
fi

echo "Downloading Place365 dataset..."
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
tar -xf places365standard_easyformat.tar
rm places365standard_easyformat.tar
cd ../../