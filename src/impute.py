# src/impute.py
import pandas as pd
from sklearn.impute import KNNImputer

# ──────────────────────────────────────────────────────
# 0.  MDL lookup table (µmol L⁻¹)  ← update if lab gives lab-specific MDLs
# ──────────────────────────────────────────────────────
MDL = {
    "nitrate_nitrite": 0.02,
    "ammonium":        0.01,
    "phosphate":       0.02,
    "silicate":        0.20,
}

# ──────────────────────────────────────────────────────
# 1.  Nutrient half-MDL fill + flag
# ──────────────────────────────────────────────────────
def fill_nutrients(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mdl in MDL.items():
        if col not in df.columns:
            continue
        flag_col = f"{col}_imputed"
        mask = df[col].isna()
        df.loc[mask, col] = mdl / 2
        df[flag_col] = mask          # True where filled
    return df

# ──────────────────────────────────────────────────────
# 2.  KNN impute remaining numeric NaNs + flag
# ──────────────────────────────────────────────────────
def knn_fill_rest(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    df = df.copy()
    # a) choose numeric columns that still have NaNs
    numeric = df.select_dtypes("number").columns
    na_cols = [c for c in numeric if df[c].isna().any()]

    if not na_cols:
        return df  # nothing to do

    imputer = KNNImputer(n_neighbors=n_neighbors)
    filled = pd.DataFrame(
        imputer.fit_transform(df[na_cols]),
        columns=na_cols,
        index=df.index,
    )

    for col in na_cols:
        flag_col = f"{col}_imputed"
        mask = df[col].isna()
        df[col] = filled[col]
        df[flag_col] = mask          # True where predicted by KNN

    return df

# ──────────────────────────────────────────────────────
# 3.  One-liner convenience
# ──────────────────────────────────────────────────────
def impute_all(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return knn_fill_rest(fill_nutrients(df), n_neighbors=k)
