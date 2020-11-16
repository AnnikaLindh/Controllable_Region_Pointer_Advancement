#!/bin/sh


echo "Creating directories..."
mkdir cg_checkpoints

echo "Downloading preprocessed data..."
gdown https://drive.google.com/uc?id=1aTHBFW0XFDWsjSc4SOETapszi5pLDctc

echo "Unpacking preprocessed data..."
unzip -q data.zip

echo "Finished setup!"
