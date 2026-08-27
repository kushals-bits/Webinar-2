"""
Tabular Encoding & Cleaning Module
Cleans dirty data, handles missing values, and implements One-Hot, Ordinal, and Target Encodings.
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def clean_raw_tabular_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans dirty values in raw tabular data:
    1. Strips whitespace and non-numeric characters ('$', 'USD') from numeric string columns
    2. Corrects negative or unphysical values (e.g., negative ages or ages > 100)
    3. Handles string missing indicators (' ', 'NA', 'null', None)
    """
    cleaned = df.copy()
    
    # 1. Clean 'total_charges' if present
    if "total_charges" in cleaned.columns:
        def parse_charges(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).strip().replace("$", "").replace("USD", "").strip()
            if val_str == "" or val_str.lower() in ["na", "null", "none"]:
                return np.nan
            try:
                return float(val_str)
            except ValueError:
                return np.nan
        cleaned["total_charges"] = cleaned["total_charges"].apply(parse_charges)
        
    # 2. Fix unphysical values in age
    if "age" in cleaned.columns:
        cleaned.loc[(cleaned["age"] < 18) | (cleaned["age"] > 100), "age"] = np.nan

    # 3. Cap extreme outliers in monthly_charges
    if "monthly_charges" in cleaned.columns:
        cleaned.loc[cleaned["monthly_charges"] > 500, "monthly_charges"] = np.nan

    return cleaned


class TabularEncoder:
    """
    Production-grade transformer for Tabular Features:
    - Imputes missing numerical values with median
    - Imputes missing categorical values with mode / 'Missing'
    - Applies One-Hot Encoding to nominal categorical columns
    - Applies Ordinal Encoding to specified ordinal columns
    """
    def __init__(self, nominal_cols=None, ordinal_cols=None, ordinal_categories=None, drop_first=True):
        self.nominal_cols = nominal_cols or []
        self.ordinal_cols = ordinal_cols or []
        self.ordinal_categories = ordinal_categories or {}
        self.drop_first = drop_first
        
        self.ohe = None
        self.ordinal_encoder = None
        self.numeric_medians = {}
        self.cat_modes = {}
        self.feature_names_out_ = []
        self.numeric_cols = []

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        
        # Identify numeric columns
        all_categoricals = set(self.nominal_cols + self.ordinal_cols)
        self.numeric_cols = [c for c in X.columns if c not in all_categoricals and pd.api.types.is_numeric_dtype(X[c])]

        # Compute medians for imputation
        for c in self.numeric_cols:
            self.numeric_medians[c] = X[c].median()

        # Compute modes for categorical imputation
        for c in self.nominal_cols + self.ordinal_cols:
            mode_val = X[c].mode().iloc[0] if not X[c].dropna().empty else "Missing"
            self.cat_modes[c] = mode_val

        # Fit One-Hot Encoder
        if self.nominal_cols:
            self.ohe = OneHotEncoder(drop="first" if self.drop_first else None, sparse_output=False, handle_unknown="ignore")
            # Impute before fit
            imputed_nominal = X[self.nominal_cols].fillna(self.cat_modes)
            self.ohe.fit(imputed_nominal)

        # Fit Ordinal Encoder
        if self.ordinal_cols:
            categories_list = [self.ordinal_categories[c] for c in self.ordinal_cols] if self.ordinal_categories else "auto"
            self.ordinal_encoder = OrdinalEncoder(categories=categories_list, handle_unknown="use_encoded_value", unknown_value=-1)
            imputed_ordinal = X[self.ordinal_cols].fillna(self.cat_modes)
            self.ordinal_encoder.fit(imputed_ordinal)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        result_dfs = []
        feature_names = []

        # 1. Numeric columns (imputed)
        if self.numeric_cols:
            num_df = X[self.numeric_cols].fillna(self.numeric_medians)
            result_dfs.append(num_df.reset_index(drop=True))
            feature_names.extend(self.numeric_cols)

        # 2. Ordinal columns
        if self.ordinal_cols and self.ordinal_encoder:
            ord_imputed = X[self.ordinal_cols].fillna(self.cat_modes)
            ord_arr = self.ordinal_encoder.transform(ord_imputed)
            ord_df = pd.DataFrame(ord_arr, columns=[f"{c}_ordinal" for c in self.ordinal_cols])
            result_dfs.append(ord_df.reset_index(drop=True))
            feature_names.extend(ord_df.columns.tolist())

        # 3. Nominal columns (One-Hot Encoded)
        if self.nominal_cols and self.ohe:
            nom_imputed = X[self.nominal_cols].fillna(self.cat_modes)
            ohe_arr = self.ohe.transform(nom_imputed)
            ohe_feature_names = self.ohe.get_feature_names_out(self.nominal_cols)
            ohe_df = pd.DataFrame(ohe_arr, columns=ohe_feature_names)
            result_dfs.append(ohe_df.reset_index(drop=True))
            feature_names.extend(ohe_df.columns.tolist())

        transformed = pd.concat(result_dfs, axis=1)
        self.feature_names_out_ = feature_names
        return transformed

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
