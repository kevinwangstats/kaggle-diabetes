"""
Script for data preprocessing utilities.
Handles automated feature grouping and sklearn pipeline construction.
"""

import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def get_feature_groups(df, target_col, id_col, unique_threshold=10):
    """
    Identifies column groups for preprocessing.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Target column name.
        id_col (str): ID column name to exclude.
        unique_threshold (int): Threshold for treating numeric cols as ordinal/categorical.
        
    Returns:
        tuple: (numeric_continuous_cols, numeric_ordinal_cols, string_categorical_cols)
    """
    # All features excluding target and ID (if present in df)
    # Be robust if target_col is absent (e.g. test data)
    cols_to_exclude = [id_col]
    if target_col in df.columns:
        cols_to_exclude.append(target_col)
        
    features = [c for c in df.columns if c not in cols_to_exclude]
    
    # Identify string/object columns which are strictly categorical
    string_categorical_cols = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Identify numeric columns
    numeric_cols = df[features].select_dtypes(include=['number']).columns.tolist()
    
    numeric_continuous_cols = []
    numeric_ordinal_cols = []
    
    for col in numeric_cols:
        # Check cardinality
        if df[col].nunique() < unique_threshold:
            numeric_ordinal_cols.append(col)
        else:
            numeric_continuous_cols.append(col)
            
    logger.info(f"Continuous Numeric Features ({len(numeric_continuous_cols)}): {numeric_continuous_cols}")
    logger.info(f"Ordinal Numeric Features ( < {unique_threshold} unique) ({len(numeric_ordinal_cols)}): {numeric_ordinal_cols}")
    logger.info(f"Categorical String Features ({len(string_categorical_cols)}): {string_categorical_cols}")
            
    return numeric_continuous_cols, numeric_ordinal_cols, string_categorical_cols

def build_pipeline(num_continuous_cols, num_ordinal_cols, cat_string_cols):
    """
    Builds the sklearn preprocessing pipeline.
    
    Returns:
        ColumnTransformer: The preprocessing step of the pipeline.
    """
    # 1. Continuous Numeric -> Standard Scaler
    pipe_continuous = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # 2. Ordinal Numeric -> Ordinal Encoder
    pipe_ordinal = Pipeline([
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # 3. String Categorical -> OneHot Encoder
    pipe_categorical = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', pipe_continuous, num_continuous_cols),
            ('ordinal', pipe_ordinal, num_ordinal_cols),
            ('categorical', pipe_categorical, cat_string_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    return preprocessor
