# PROTAC-STAN

This is the official codebase of the paper: "Interpretable PROTAC degradation prediction with structure-informed deep ternary attention framework"

## Overview

This study introduces *PROTAC-STAN*, a **structure-informed deep ternary attention network (STAN)** framework for interpretable PROTAC degradation prediction. PROTAC-STAN represents PROTAC molecules across **atom, molecule, and property hierarchies** and incorporates **structure information** for POIs and E3 ligases using a protein language model infused with structural data. Furthermore, it simulates interactions among three entities via a **novel ternary attention network** tailored for the PROTAC system, providing unprecedented insights into the degradation mechanism.

<img src="assets/PROTAC-STAN.png" alt="Overview of PROTAC-STAN" width="100%">

## Installation

1. Create Conda environment
``` shell
conda create -n PROTAC-STAN python=3.11.5
conda activate PROTAC-STAN
```
2. Install Pytorch
```shell
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
3. Install other essential packages
```shell
rdkit
pyg
pandas
toml
...
```

> [!TIP]
> See `protac-stan.yml` for full requriements.

## Datasets
### PROTAC-DB
The original data can be accessed at [PROTAC-DB](http://cadd.zju.edu.cn/protacdb/). 

> [!NOTE]
> We also provide the PROTAC-DB 2.0 in `data/PROTAC-DB2` folder for your convenience.

### PROTAC-fine
We enrich degradation information to the [PROTAC-DB 2.0](https://academic.oup.com/nar/article/51/D1/D1367/6775390) and construct a refined PROTAC dataset named PROTAC-fine. The data are stored in `data/PROTAC-fine` folder.

## Directory instructions

### Training and inference

```txt
.
├── config.toml
├── data
├── data_loader.py
├── data.py
├── inference.py
├── main.py
├── model.py
├── saved_models
└── tan.py
```

### Custom data preparation

```txt
.
├── data
│   ├── custom
├── esm_embed
│   ├── get_embed_s.py
│   ├── model
│   └── README.md
├── prepare_data.ipynb
```

## Training

We have prepared the PROTAC-fine dataset in directory `data/PROTAC-fine`. 

To train the PROTAC-STAN model from scratch, run the following script:

```shell
python main.py
```
Evaluation results of PROTAC-STAN and baselines on test set considering data leakage:

<img src="assets/results.png" alt="Evaluation results of PROTAC-STAN and baselines on test set considering data leakage" width="70%">

## Inference

`inference.py` leverage PROTAC-STAN as a powerful tool to perform interpretable PROTAC degradation prediction.

1. Prepare your customed data following `prepare_data.ipynb`
2. Predict your data:
```shell
# Usage: python inference.py [-h] [--root ROOT] [--name NAME] [--save_att]
python inference.py --root 'data/custom' --name 'custom'
```

You may use attention maps to take further anaysis, here are our examples:

> [!TIP]
> You may use Python packages like [matplotlib](https://matplotlib.org/stable/), [RDKit](https://www.rdkit.org/), Visualization software like [Maestro](https://www.schrodinger.com/platform/products/maestro/), [PyMOL](https://www.pymol.org/) and so on.

<details>
<summary> 3D and 2D attention map visualization </summary>
<img src="assets/2d+3d.png" alt="3D and 2D attention map visualization" width="100%">
</details>

<details open>
<summary> Molecule and complex visualization </summary>
<img src="assets/complex.png" alt="Molecule and complex visualization" width="100%">
</details>

## Citation
