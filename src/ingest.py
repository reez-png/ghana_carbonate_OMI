# src/ingest.py
from pathlib import Path
from typing import Dict, List
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

ROOT         = Path(__file__).resolve().parents[1]
RAW_DIR      = ROOT / "data" / "raw"
INTERIM_DIR  = ROOT / "data" / "interim"

# ──────────────────────────────────────────────────────────
# 1.  Universal schema for every sheet (cruise)
# ──────────────────────────────────────────────────────────
SCHEMA = pa.DataFrameSchema(
    {
        # alphanumeric, 1–4 chars, e.g. S11, J110
        "station": Column(
            str,
            Check.str_matches(r"^[A-Z]\d{1,3}$"),   # optional regex guard
            nullable=False,
            coerce=True,
        ),
        "date":       Column(str),                     # we'll convert later
        "time":       Column(str),
        "lat":        Column(float, Check.between(-10, 10)),
        "lon":        Column(float, Check.between(-5, 5)),
        "loc":        Column(str),
        "season":     Column(str),
        "depth_m":    Column(float, Check.ge(0)),
        "depth_desc": Column(str),
        "sample_id":  Column(str),
        "temp_wat":   Column(float, Check.between(0, 40)),
        "temp_in_lb": Column(float, Check.between(0, 40)),
        "sal_wat":    Column(float, Check.between(0, 40)),
        "pres":       Column(float, Check.ge(0)),
        "ph_lb":      Column(float, Check.between(7.0, 8.5)),
        "ta":         Column(float, Check.ge(1800)),
        "ph":         Column(float, Check.between(7.0, 8.5)),
        "nitrate_nitrite": Column(
            float,
            Check.ge(0, ignore_na=True),   # allow blanks; still flag negatives
            nullable=True,
        ),
        "ammonium": Column(
            float,
            Check.ge(0, ignore_na=True),
            nullable=True,
        ),
        "phosphate": Column(
            float,
            Check.ge(0, ignore_na=True),
            nullable=True,
        ),
        "silicate":  Column(
            float,
            Check.ge(0, ignore_na=True),
            nullable=True,
        ),
        "chl":       Column(
            float,
            Check.ge(0, ignore_na=True),
            nullable=True,
        ),
        "o2":        Column(
            float,
            Check.ge(0, ignore_na=True),
            nullable=True
        )
    },
    coerce=True,        # cast types automatically
    strict=False,       # ignore extra cols for now
)

# ──────────────────────────────────────────────────────────
# 2.  Helper: tidy a *single* sheet DataFrame
# ──────────────────────────────────────────────────────────
def _clean_sheet(df: pd.DataFrame, cruise_name: str) -> pd.DataFrame:
    # a. uniform lowercase snake_case headers
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(" ", "_")
    )

    # b. basic type fixes (string → datetime)
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )

    # c. add cruise label
    df["cruise"] = cruise_name

    # d. run schema validation (raises if any row violates checks)
    SCHEMA.validate(df, lazy=True)

    return df

# ──────────────────────────────────────────────────────────
# 3.  Public function: read *all* sheets in one call
# ──────────────────────────────────────────────────────────
def load_excel_multisheet(path: Path) -> pd.DataFrame:
    """
    Read every worksheet in an Excel workbook, validate, and concatenate
    them into one tidy DataFrame.
    """
    # Pandas returns Dict[str, DataFrame] when sheet_name=None
    all_sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")

    cleaned = []
    for sheet_name, df in all_sheets.items():
        print(f"▶ Sheet '{sheet_name}': {df.shape} rows × {df.shape[1]} cols")  # ⚑

        if df.dropna(how="all").empty:
            print("  ↳ sheet is blank; skipping")  # ⚑
            continue

        try:
            cleaned.append(_clean_sheet(df, cruise_name=sheet_name))
        except pa.errors.SchemaErrors as err:  # ⚑
            print(f"  ↳ validation failed:\n{err.failure_cases.head()}")  # ⚑
            # keep looping so other sheets still have a chance
            continue

    if not cleaned:
        raise ValueError(f"No sheet in {path.name} passed QC")

    return pd.concat(cleaned, ignore_index=True)

# ──────────────────────────────────────────────────────────
# 4.  CLI convenience wrapper
# ──────────────────────────────────────────────────────────
def ingest_book(excel_path: Path) -> Path:
    df = load_excel_multisheet(excel_path)
    out = INTERIM_DIR / f"{excel_path.stem}.parquet"
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"✔ Saved {len(df)} rows → {out}")
    return out