# TriM-DDS

## Overview
![model](https://github.com/AnQi-87/TriM-DDS/blob/main/TriM-DDS.png)

## Environments
### GPU
```bash
CUDA 12.4
```
### conda
```bash
conda create -n TriM-DDS python=3.10 -y
conda activate TriM-DDS
conda install scipy -y
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge rdkit
pip install torch-geometric==2.5.3 -i https://pypi.org/simple/
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.5.1%2Bcu124.html
pip install seaborn -i https://pypi.org/simple/
```
## Train
```bash
python trim_train.py
```

