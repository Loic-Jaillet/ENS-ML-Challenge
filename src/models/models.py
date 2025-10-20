import joblib
import os
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from hyperparameters import RSF_PARAMS, GBS_PARAMS

def build_model(model_name):
    """
    Initialize model with the right hyperparameters.
    model_name: 'rsf' or 'gbs'
    """
    if model_name.lower() == "rsf":
        model = RandomSurvivalForest(**RSF_PARAMS)
    elif model_name.lower() == "gbs":
        model = GradientBoostingSurvivalAnalysis(**GBS_PARAMS)
    else:
        raise ValueError("Unknown model_name. Choose 'rsf' or 'gbs'.")
    return model


def train_and_save_model(model_name, X_train, y_train, artifacts_dir="../../artifacts"):
    """
    Train the given model type and save it to artifacts/.
    """
    # Build model
    model = build_model(model_name)

    print(f"Training {model_name.upper()} model...")
    if model_name.lower() == 'rsf':
        print(f"RSF parameters: {RSF_PARAMS}")
    elif model_name.lower() == 'gbs':
        print(f"GBS parameters: {GBS_PARAMS}")
        
    model.fit(X_train, y_train)
    print(f"{model_name.upper()} training complete.")

    # Save model
    os.makedirs(artifacts_dir, exist_ok=True)
    save_path = os.path.join(artifacts_dir, f"{model_name}_model.pkl")
    joblib.dump(model, save_path)
    print(f"Model saved at: {save_path}")

    return model

def load_model(model_name, artifacts_dir="../../artifacts"):
    """
    Load a saved model from artifacts/.
    """
    load_path = os.path.join(artifacts_dir, f"{model_name}_model.pkl")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No saved model with name {model_name} found at: {load_path}")
    
    model = joblib.load(load_path)
    print(f"Loaded {model_name.upper()} model from: {load_path}")
    return model


