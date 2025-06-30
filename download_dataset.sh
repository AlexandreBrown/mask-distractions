#!/bin/bash
DATA_DIR=./datasets/

if [ -d "${DATA_DIR}places365_standard" ] && [ "$(ls -A "${DATA_DIR}places365_standard")" ]
then
  echo "Place365 dataset already present in $DATA_DIR skipping download."
else
  echo "Downloading Place365 dataset..."
  mkdir -p "$DATA_DIR"
  wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar -O "${DATA_DIR}places365standard_easyformat.tar"
  tar -xf "${DATA_DIR}places365standard_easyformat.tar" -C "$DATA_DIR"
fi