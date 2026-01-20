# scPertCRL (Simple Version)

## 📌 Overview

**scPertCRL** is a causal representation learning framework for **single-cell perturbation prediction**.
It integrates **variational autoencoders (VAE)** with **directed acyclic graph (DAG) constraints** to explicitly model causal dependency structures among genes or features, enabling more interpretable and generalizable perturbation effect prediction.

> ⚠️ **Note**
> This repository provides a **simplified version** of scPertCRL for reproducibility and demonstration purposes.
> A **complete version**, including full model variants, additional ablation settings, and extended documentation, will be **publicly released after paper acceptance**.

---

## 📁 Directory Structure

```
scPertCRL/
├── config/                    # Experiment configuration files
│   ├── chem.yaml              # Chemical perturbation settings
│   └── gene.yaml              # Gene perturbation settings
├── DAGPert/                   # Core DAG-aware perturbation modeling code
│   ├── model/
│   │   ├── model_base.py      # Base model definitions
│   │   ├── model_component.py # Neural network components
│   │   ├── model_ge.py        # Gene / perturbation embedding modules
│   │   └── model_utils.py     # Helper functions
│   ├── data_preprocess.py     # Data loading and preprocessing
│   ├── eval.py                # Evaluation metrics and workflows
│   ├── train_model_addDAGVAE.py # Training logic for DAG-VAE
│   └── utils.py               # General utilities
├── data/                      # (Optional) Default data directory
├── exps/                      # Experiment entry points
│   ├── eval.py                # Evaluation runner
│   └── main_addDAGVAE.py      # Main training script
├── model/                      # (Optional) Default data directory
└── README.md
```

---

## 🛠️ Installation

### Requirements

* Python ≥ 3.9
* torch 2.3.0
* torch_geometric 2.5.2
* transformers 4.46.1
* scanpy 1.10.1
* numpy 1.26.4
* pandas 2.2.2
* scipy 1.13.0
* scikit-learn 1.4.2
* networkx 3.2.1
* geomloss 0.2.6
* torchmetrics 1.7.2
* matplotlib 3.8.4
* rdkit 2024.3.5
* cell-gears 0.1.2
* tqdm 4.66.2
* Other standard dependencies

> We recommend creating a clean environment following the setup used in
> **cell-eval**: [https://github.com/ArcInstitute/cell-eval](https://github.com/ArcInstitute/cell-eval)

---

## 🚀 Quick Start

### 0. Download Example Data (Optional)
You can download example datasets for testing or demonstration from Google Drive:

[Google Drive Link](https://drive.google.com/drive/folders/EXAMPLE_LINK)

### 1. Configure Experiments

Edit configuration files in `config/` to specify:

* perturbation types and splits
* model hyperparameters
* data paths


### 2. Train the Model

Run training with:

```bash
python exps/main_addDAGVAE.py --config config/gene.yaml
```

### 3. Evaluate Performance

Evaluate trained models via:

```bash
python exps/eval.py --path ./output/
```

---

## 🧩 Core Components

* **DAG-aware VAE**
  Combines latent-variable modeling with explicit DAG constraints to encourage causal structure discovery.

* **Perturbation-aware Encoding**
  Supports gene and chemical perturbations through learned embeddings.

* **Flexible Evaluation Pipeline**
  Designed to integrate with standard single-cell perturbation benchmarks.

---

## 📄 About This Release

The **full version** will be released **upon acceptance of the corresponding manuscript**.
