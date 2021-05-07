#!/bin/sh


echo "Creating directories..."
mkdir cg_checkpoints

echo "Downloading preprocessed data..."
gdown https://drive.google.com/uc?id=18fSwjw_F2aL-nDpQouEJ9Mh_cC12SyW9

echo "Unpacking preprocessed data..."
unzip -q data.zip

echo "Finished setup!"
