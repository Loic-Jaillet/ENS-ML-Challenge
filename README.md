# QRT-2025-Challenge

This repository contains some of my solutions for the **QRT ENS 2025 Overall Survival Prediction for Patients Diagnosed with Myeloid Leukemia** challenge. The best solution giving a top 3% ranking in the competition.

The goal is to predict patient survival using clinical and molecular features, leveraging Random Survival Forests (RSF) and Gradient Boosted Survival (GBS) models.

More details about the challenge can be found here: [QRT ENS 2025 Competition](https://challengedata.ens.fr/participants/challenges/162/)


---

## Folder Structure

- `notebooks/` : Jupyter notebooks testing the code.  
- `src/` : Source code
    - `models/` : Model building, training, and evaluation scripts
    - `utils/` : Helper functions (feature evaluation...)
- `artifacts/` : Saved trained models, scalers, and encoders (**ignores trained models if size too big in Github**)  
- `Data/` : Raw datasets

---

## Installation / Requirements

- Python 3.10+  
- Required packages (can be installed via `requirements.txt`):
    - pandas, numpy, scikit-learn
    - joblib, scipy
    - lifelines (for RSF)
    - matplotlib, seaborn (optional for visualization)

```bash
pip install -r requirements.txt

