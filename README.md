# QRT-2025-Challenge

This repository contains some of my solutions for the **QRT ENS 2025 Overall Survival Prediction for Patients Diagnosed with Myeloid Leukemia** challenge. The best solution giving a top 3% ranking in the competition.

The goal is to predict patient survival using clinical and molecular features.
I leveraged Random Survival Forests (RSF) and Gradient Boosted Survival (GBS) models for the best performance in Survival Analysis.

More details about the challenge can be found here: [QRT ENS 2025 Competition](https://challengedata.ens.fr/participants/challenges/162/)

Others details can be found in the `Benchmark` notebook.

The `challenge_v1` notebook is my first attempt at the competition with all details and the whole code and some explainations and curves to understand parts of the process.

The `challenge_v2` is the final version with all details, it gives the best results.

A much easier and faster way to use the code is shown in the `notebooks/` folder from the modular architecture of the code.

---

## Folder Structure

- `notebooks/` : Jupyter notebooks testing the code.  
- `src/` : Source code, utils, preprocessing for all datasets
    - `models/` : Model building, training, and evaluation scripts
- `artifacts/` : Saved trained models, scalers, and encoders (**ignores trained models if size too big in Github**)  
- `Data/` : Raw datasets

---

## Installation / Requirements

- Python 3.10+  
- Required packages (can be installed via `requirements.txt`):
    - pandas, numpy, scikit-learn
    - joblib, scipy
    - scikit_survival (Survival Analysis)
    - matplotlib, seaborn (for visualization)

```bash
pip install -r requirements.txt

