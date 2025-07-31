# ───────────────────────────────  src/omp.py  ───────────────────────────────
"""
Optimum Multiparameter (OMP) solver
-----------------------------------

• Reads end-member means + 1-σ uncertainties from data/metadata/endmembers.yml
• Solves bounded least-squares for every sample row
• Returns the original DataFrame with *_frac columns appended
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import lsq_linear


# ---------------------------------------------------------------------------
# 0.  Read YAML → means (DataFrame)  &  sigmas (Series of 1-σ per tracer)
# ---------------------------------------------------------------------------
def load_swts(yml: Path | str = "data/metadata/endmembers.yml"):
    """
    Parameters
    ----------
    yml : path to YAML file with structure::

        LUW:
          CT:   23.4
          CT_sd: 0.2
          S:    35.9
          S_sd: 0.05
          O2:   210
          O2_sd: 3
        ...

    Returns
    -------
    means  : DataFrame  [SWT × tracer]   e.g. LUW–OOTW–RSW rows
    sigmas : Series     [tracer]         average 1-σ per tracer
    """
    with open(yml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    full = pd.DataFrame(raw).T.astype(float)

    mean_cols = [c for c in full.columns if not c.endswith("_sd")]
    sd_cols   = [c for c in full.columns if c.endswith("_sd")]

    means  = full[mean_cols]                                         # SWT × tracer
    sigmas = (
        full[sd_cols]
        .rename(columns=lambda c: c.replace("_sd", ""))              # CT_sd → CT
        .mean()                                                      # 1-σ vector
    )

    return means, sigmas


# ---------------------------------------------------------------------------
# 1.  Solve one sample (Series) → fractions (Series)
# ---------------------------------------------------------------------------
def solve_single(sample: pd.Series,
                 means: pd.DataFrame,
                 sigmas: pd.Series) -> pd.Series:
    """
    Returns a Series of mixing fractions for the water-mass rows in `means`.
    If the system is under-determined (<2 valid tracers) or ill-conditioned,
    returns NaNs so the caller can flag or drop that sample.
    """
    # keep only tracer columns that are numeric in this sample
    good = sample[means.columns].dropna()
    if good.size < 2:                                      # need ≥2 constraints
        return pd.Series(np.nan, index=means.index)

    A = (means[good.index] / sigmas[good.index]).T.to_numpy(dtype=float)
    b = (good / sigmas[good.index]).to_numpy(dtype=float)

    try:
        res = lsq_linear(A, b, bounds=(0, 1), method="bvls")
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=means.index)

    f = res.x
    f = f / f.sum() if f.sum() > 0 else np.nan             # normalise Σf = 1
    return pd.Series(f, index=means.index)


# ---------------------------------------------------------------------------
# 2.  Apply to an entire DataFrame
# ---------------------------------------------------------------------------
def solve_df(df: pd.DataFrame,
             means: pd.DataFrame,
             sigmas: pd.Series) -> pd.DataFrame:
    """
    Adds one *_frac column per SWT and returns the expanded DataFrame.
    """
    frac = df.apply(solve_single, axis=1, args=(means, sigmas))
    return pd.concat([df.reset_index(drop=True),
                      frac.add_suffix("_frac")], axis=1)
# ────────────────────────────────────────────────────────────────────────────
