import pandas as pd
from sklearn.inspection import permutation_importance



# feature importance for survival models

def compute_permutation_importance(model, X, y, n_repeats=5, random_state=42, top_n=20):
    """
    Compute permutation feature importance for any fitted model.

    Parameters:
        model: fitted model (e.g. RSF, GradientBoostingSurvivalAnalysis)
        X: pd.DataFrame of features
        y: target (array-like)
        n_repeats: number of shuffles for importance estimation
        random_state: reproducibility
        top_n: number of top features to display

    Returns:
        pd.DataFrame with mean and std of feature importances
    """
    # For survival models, model.predict(X) should output risk or survival score
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state
    )

    importance_df = pd.DataFrame({
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std
    }, index=X.columns)

    importance_df = importance_df.sort_values(
        by="importances_mean", ascending=False
    )

    if top_n:
        return importance_df.head(top_n)
    return importance_df
