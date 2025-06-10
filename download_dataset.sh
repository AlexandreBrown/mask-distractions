#!/bin/bash
DATA_DIR=./datasets/

if [ -d "${DATA_DIR}places365_standard" ] && [ "$(ls -A "${DATA_DIR}places365_standard")" ]
then
  echo "Place365 dataset already present in $DATA_DIR skipping download."
else
  echo "Downloading Place365 dataset..."
  mkdir -p "$DATA_DIR"
  cd "$DATA_DIR"
  wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
  tar -xf places365standard_easyformat.tar
  rm places365standard_easyformat.tar
  cd ../../
fi
