#!/usr/bin/env bash
mkdir pretrained_resnext-101
cd pretrained_resnext-101
echo "Downloading ResNext-101 pretrained model...\n"
wget http://data.dmlc.ml/models/imagenet/resnext/101-layers/resnext-101-64x4d-symbol.json
wget http://data.dmlc.ml/models/imagenet/resnext/101-layers/resnext-101-64x4d-0000.params
wget http://data.dmlc.ml/models/imagenet/resnext/synset.txt
echo "\n"

echo "Downloading data...\n"
mkdir data
cd ../data




echo "\n"