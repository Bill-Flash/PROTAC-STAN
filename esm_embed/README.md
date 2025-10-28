# ESM-S Embedder

We have developed the ESM Embedder to processing protein sequence features into implicit structural representations.

## Directory structure

```txt
.
├── get_embed_s.py
├── model
│   ├── esm2_t33_650M_UR50D.pt
│   └── esm_650m_s.pth
└── README.md
```

The ESM model weights `esm2_t33_650M_UR50D.pt` can be found [here](https://github.com/facebookresearch/esm), and the structure-informed ESM model weights `esm_650m_s.pth` can be found [here](https://github.com/DeepGraphLearning/esm-s).

> [!IMPORTANT]
> Please ensure you have downloaded the model weights!

## Requirements

We need a new conda environment named `esm+` here. 

```shell
conda create -n esm+ python=3.8.18
conda install torchdrug pytorch=1.12.1 cudatoolkit=11.6 -c milagraph -c pytorch-lts -c pyg -c conda-forge
conda install easydict pyyaml -c conda-forge
```
## Updated conda 
```shell
conda create -n esm+ python=3.10 -y
conda activate esm+

# 1️⃣ 安装 PyTorch (官方 CUDA 12.1 版)
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3️⃣ 安装 PyTorch Geometric 套件（PyG 官方 wheel）
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 2️⃣ 安装 TorchDrug（pip 官方源即可，0.2.1 支持 PyTorch 2.x）
pip install torchdrug

# 4️⃣ 其他依赖
pip install easydict pyyaml tqdm
conda install numpy=1.26 -y
pip install --force-reinstall "scipy<1.13"
```

> [!TIP]
> We develop ESM-S Embedder based on ESM-S, [click](https://github.com/DeepGraphLearning/esm-s) for details.

## Usage

```shell
# python get_embed_s.py [-h] [--root ROOT]
python get_embed_s.py --root '../data/custom'