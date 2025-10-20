# QRT-2025-Challenge

This repository contains some of my solutions for the **QRT ENS 2025 Overall Survival Prediction for Patients Diagnosed with Myeloid Leukemia** challenge. The best solution giving a top 3% ranking in the competition.

The goal is to predict patient survival using clinical and molecular features, leveraging Random Survival Forests (RSF) and Gradient Boosted Survival (GBS) models.

More details about the challenge can be found here: [QRT ENS 2025 Competition](https://challengedata.ens.fr/participants/challenges/162/)


---

## Folder Structure

- `notebooks/` : Jupyter notebooks for exploration, experimentation, and visualization.  
- `src/` : Source code
    - `data/` : Preprocessing and feature engineering scripts
    - `models/` : Model building, training, and evaluation scripts
    - `utils/` : Helper functions (paths, metrics, feature evaluation)
- `artifacts/` : Saved trained models, scalers, and encoders (**ignored by Git due to size**)  
- `data/` : Raw and processed datasets (**ignored if too large**)  

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

