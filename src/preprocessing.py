
"""
data_preprocessing.py

Contains preprocessing and feature engineering functions for clinical 
and molecular datasets in the Myeloid Leukemia Survival Prediction project.
"""

import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib

def preprocess_clinical(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the clinical dataset by cleaning missing values and 
    engineering cytogenetic-related features.
    """
    df = clinical_df.copy()

    # Handle missing CYTOGENETICS
    df['CYTOGENETICS'] = df['CYTOGENETICS'].fillna('NA')

    # Gender: infer from CYTOGENETICS (XY → 1, XX → 0, unknown → 0.5)
    df['GENDER'] = df['CYTOGENETICS'].apply(
        lambda x: 1 if re.search(r'xy', x, re.IGNORECASE)
        else (0 if re.search(r'xx', x, re.IGNORECASE) else 0.5)
    )

    # Number of abnormalities (translocations, deletions, etc.)
    df['NUM_ABNORMALITIES'] = df['CYTOGENETICS'].apply(
        lambda x: len(re.findall(r'(t\(.+?\)|del\(.+?\)|dup\(.+?\)|inv\(.+?\)|\+\d+|-\d+)', x))
    )

    # Chromosome count deviation from 46
    df['CHROMOSOME_DIFF'] = df['CYTOGENETICS'].apply(
        lambda x: abs(int(re.search(r'^\d+', x).group()) - 46)
        if re.search(r'^\d+', x) else 0
    )

    return df


def enrich_with_molecular_features(clinical_df: pd.DataFrame, molecular_df: pd.DataFrame, encoder, dbscan: DBSCAN) -> pd.DataFrame:
    """
    Merges molecular-level features (mutations, chromosomal data, clustering, encoded categorical features)
    into the clinical dataset.
    """
    df = clinical_df.copy()

    # Merge number of mutations
    tmp = molecular_df.groupby('ID').size().reset_index(name='Nb mut')
    df = df.merge(tmp, on='ID', how='left').fillna({'Nb mut': 0})

    # Add specific chromosomal abnormalities
    df['MONOSOMY 9'] = df['CYTOGENETICS'].str.contains('-9').fillna(False).map({True: 1, False: 0})
    df['MONOSOMY 7'] = df['CYTOGENETICS'].str.contains('-7').fillna(False).map({True: 1, False: 0})

    # Aggregate molecular stats
    for col, name in [('VAF', 'sum_VAF'), ('DEPTH', 'sum_depth')]:
        tmp = molecular_df.groupby('ID')[col].sum().reset_index()
        tmp.columns = ['ID', name]
        df = df.merge(tmp, on='ID', how='left').fillna({col: 0})

    # X chromosome mutation count
    molecular_df['X_mutation'] = molecular_df['CHR'].apply(lambda x: 1 if x == 'X' else 0)
    mol_sum_X = molecular_df.groupby('ID')['X_mutation'].sum().reset_index().rename(columns={'X_mutation': 'sum_X'})
    df = df.merge(mol_sum_X, on='ID', how='left').fillna({'sum_X': 0})

    # Clustering genomic coordinates
    coords = molecular_df[['CHR', 'START', 'END']].copy()
    coords['CHR'] = coords['CHR'].dropna().apply(lambda x: 23.0 if x == 'X' else float(x))
    coords['START'] = coords['START'].fillna(coords['START'].mean())
    coords['END'] = coords['END'].fillna(coords['END'].mean())
    coords['CHR'] = coords['CHR'].fillna(coords['CHR'].mean())

    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    clusters = dbscan.fit_predict(coords_scaled)
    molecular_df['Cluster'] = clusters

    # Protein change parsing
    molecular_df['PROTEIN_CHANGE'] = molecular_df['PROTEIN_CHANGE'].fillna('NA')
    protein_features = molecular_df['PROTEIN_CHANGE'].apply(parse_protein_change).apply(pd.Series)

    for col in protein_features.columns:
        if protein_features[col].dtype in ('int64', 'float64'):
            protein_features[col] = protein_features[col].fillna(protein_features[col].median())
        else:
            protein_features[col] = protein_features[col].fillna('NA')

    molecular_df = pd.concat([molecular_df, protein_features], axis=1)
    molecular_df = molecular_df.replace('NA', None)

    # Encoding categorical features
    categorical_cols = ['CHR', 'GENE', 'EFFECT', 'REF', 'ALT', 'Cluster', 'original_aa', 'mutant_aa']
    encoded_features = encoder.transform(molecular_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    encoded_df['ID'] = molecular_df['ID']

    encoded_df = pd.concat([encoded_df, protein_features.drop(columns=['original_aa', 'mutant_aa'])], axis=1)
    encoded_group = encoded_df.groupby('ID').sum().reset_index()

    # Merge encoded features with clinical
    df = df.merge(encoded_group, on='ID', how='left')

    # Impute missing values for numeric columns with <12% missingness
    for col in df.columns[df.isna().mean() < 0.12]:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())

    return df


def parse_protein_change(protein_str: str) -> dict:
    """
    Parses the PROTEIN_CHANGE column into structured numeric and categorical features.
    """
    match = re.match(r'p\.([A-Z*])(\d+)(fs\*\d*|\*|\w*)', protein_str)

    if match:
        original_aa, position, mutation_type = match.groups()
        return {
            "position": int(position),
            "original_aa": original_aa,
            "mutant_aa": mutation_type if mutation_type not in ["", "*"] and 'fs' not in mutation_type else "stop",
            "frameshift": 1 if "fs" in mutation_type else 0,
            "stop_gain": 1 if "*" in mutation_type else 0,
            "stop_distance": int(mutation_type.split('*')[-1]) if "fs*" in mutation_type and mutation_type.split('*')[-1] != '' else 0,
            "deletion": 1 if "del" in protein_str else 0,
            "insertion": 1 if ("ins" in protein_str or "PTD" in protein_str) else 0
        }

    return {
        "position": None, "original_aa": None, "mutant_aa": None,
        "frameshift": 0, "stop_gain": 0, "stop_distance": 0,
        "deletion": 0, "insertion": 1 if 'PTD' in protein_str else 0
    }


def preprocess_test_data(clinical_test_df, molecular_test_df, encoder, dbscan):
    """
    Full preprocessing pipeline for test data.
    """
    clinical_proc = preprocess_clinical(clinical_test_df)
    X_test = enrich_with_molecular_features(clinical_proc, molecular_test_df, encoder, dbscan)
    return X_test




def align_and_transform(df: pd.DataFrame, X_columns: list, imputer, scaler) -> pd.DataFrame:
    """
    Align test dataframe to training columns, impute missing values, and scale features.
    
    Parameters:
        df: Test dataframe
        X_columns: List of columns from training set
        imputer: Fitted KNNImputer
        scaler: Fitted StandardScaler
    
    Returns:
        Preprocessed dataframe ready for model input
    """
    # Drop extra columns
    df = df.drop(columns=[c for c in df.columns if c not in X_columns], errors='ignore')
    
    # Add missing columns
    for col in X_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training
    df = df[X_columns]
    
    # Impute
    df_imputed = imputer.transform(df)
    
    # Scale
    df_scaled = scaler.transform(df_imputed)
    
    return pd.DataFrame(df_scaled, columns=X_columns)

# utils.py
import pandas as pd
import joblib

def finalize_preprocessing(
    df: pd.DataFrame,
    imputer_path : str = '../artifacts/imputer_knn.pkl',
    scaler_path: str =  '../artifacts/scaler_knn.pkl',
    columns_path: str = '../artifacts/X_columns.pkl'
) -> pd.DataFrame:
    """
    Finalizes preprocessing on a new dataset (test or validation) using
    previously saved imputer, scaler, and training columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Test dataset to preprocess.
    imputer_path : str
        Path to the saved fitted KNNImputer (joblib file).
    scaler_path : str
        Path to the saved fitted StandardScaler.
    columns_path : str
        Path to the saved training columns list.
    
    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe ready for model input.
    """
    
    # Load saved objects
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    train_columns = joblib.load(columns_path)
    
    # Drop extra columns not in training
    df = df.drop(columns=[c for c in df.columns if c not in train_columns], errors='ignore')
    
    # Add missing columns as 0
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training set
    df = df[train_columns]
    
    # Apply saved imputer
    df_imputed = imputer.transform(df)
    
    # Apply saved scaler
    df_scaled = scaler.transform(df_imputed)
    
    # Return as dataframe with proper column names
    return pd.DataFrame(df_scaled, columns=train_columns)

