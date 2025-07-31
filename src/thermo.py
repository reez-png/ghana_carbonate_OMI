# src/thermo.py
import gsw
import pandas as pd

def add_CT(df: pd.DataFrame,
           SP_col="sal_wat", T_col="temp_wat",
           p_col="pres", lon_col="lon", lat_col="lat") -> pd.DataFrame:
    """
    Append Absolute Salinity (SA) and Conservative Temperature (CT) columns.
    Expects pressure in decibars.
    """
    SA = gsw.SA_from_SP(df[SP_col], df[p_col], df[lon_col], df[lat_col])
    CT = gsw.CT_from_t(SA, df[T_col], df[p_col])
    out = df.copy()
    out["SA"] = SA
    out["CT"] = CT
    return out