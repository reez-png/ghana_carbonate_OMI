# ───────────────────────── make_profiles.py ─────────────────────────
#import sys
import re
#import math
from pathlib import Path

# Data & numerics
import pandas as pd
import numpy as np
import xarray as xr

# Plotting & mapping
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as patheffects
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Optional: for nicer bathymetry colors
try:
    import cmocean as cmo
    CMAP_BATHY = cmo.cm.deep
except ModuleNotFoundError:
    print("cmocean not installed – using 'Blues' instead.")
    CMAP_BATHY = 'Blues'

# ─────────────────────────
# 2. Paths & Directories
# ─────────────────────────
project_root   = Path(__file__).resolve().parent   # ALWAYS one level up
FIG_DIR        = project_root / "results" / "figures"
DATA_PROCESSED = project_root / "data" / "processed" / "water_co2_OMP_carbon.parquet"
DATA_BATHY     = project_root / "data" / "raw" / "ETOPO_2022_v1_30s_N90W180_bed.nc"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────
# 3.  Load processed data
# ─────────────────────────
df = pd.read_parquet(DATA_PROCESSED)
print("↪︎ Columns in processed file:", list(df.columns))

# ─────────────────────────
# 4.  Station order & summary
# ─────────────────────────
letters  = ["S", "J", "P", "A"]
nums     = sorted({int(re.search(r"\d+", s).group()) for s in df["station"].unique()})
stations = [f"{ltr}{n}" for n in nums for ltr in letters if f"{ltr}{n}" in df["station"].unique()]
cruises  = df["cruise"].unique()

# ----- build variables dict ONLY for columns that exist -------------
candidate_vars = {
    "pCO2"     : "pCO₂ (µatm)",
    "Omega_ar" : "Ωₐᵣ",
    "Omega_ca" : "ΩCa",
    "HCO3"     : "HCO₃⁻ (µmol kg⁻¹)",
    "CO3"      : "CO₃²⁻ (µmol kg⁻¹)",
    "ph"       : "pH",
    "ta"       : "Total Alkalinity (µmol kg⁻¹)",
}

variables = {k: v for k, v in candidate_vars.items() if k in df.columns}

if not variables:
    raise ValueError("None of the expected carbonate columns were found!")

summary = (
    df
    .groupby(["station", "cruise", "depth_m", "depth_desc"])[list(variables.keys())]
    .agg(["mean", "std"])
    .stack()
    .reset_index()
    .rename(columns={"level_4": "stat"})
)
summary["depth_m_neg"] = -summary["depth_m"]

# ─────────────────────────
# 4. Helper Functions
# ─────────────────────────
def bullet(var_key, var_label, df, depth_order):
    """Return a concise bullet or None if data are insufficient."""
    shallow_lbl, deep_lbl = depth_order[0], depth_order[-1]
    surf = df.query("depth_desc == @shallow_lbl").groupby("station")[var_key].median().dropna()
    deep = df.query("depth_desc == @deep_lbl").groupby("station")[var_key].median().dropna()

    if surf.empty or deep.empty:
        return None

    ratio_ser = (deep / surf).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio_ser.empty:
        return None
    ratio = ratio_ser.median()

    top_station = deep.idxmax() if ratio > 1 else deep.idxmin()
    top_val = deep.loc[top_station]
    trend_word = "increased" if ratio > 1 else "decreased"
    arrow = "↑" if ratio > 1 else "↓"

    return (
        f"• **{var_label}** {trend_word} with depth "
        f"({arrow} ratio ≈ {ratio:.4f}); "
        f"most extreme bottom value at {top_station} "
        f"({top_val:.4f})."
    )

def make_station_map(surf_df, bathy_ds, var_key, title, cbar_label, fname, vmin=None, vmax=None, cmap="coolwarm"):
    """Creates and saves a station map for a given variable."""
    # 1. Geographic bounds and projection
    pad = 0.3
    lon0, lon1 = surf_df["lon"].min() - pad, surf_df["lon"].max() + pad
    lat0, lat1 = surf_df["lat"].min() - pad, surf_df["lat"].max() + pad
    proj = ccrs.PlateCarree()

    # 2. Create figure
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection=proj)
    ax.set_extent([lon0, lon1, lat0, lat1], crs=proj)

    # 3. Plot bathymetry and land
    ax.contourf(bathy_ds.lon, bathy_ds.lat, bathy_ds, levels=[-200, 0], colors=["#d9d9d9"], transform=proj, zorder=0.5)
    ax.contour(bathy_ds.lon, bathy_ds.lat, bathy_ds, levels=[-2000, -1000, -200], colors="grey", linewidths=0.7, linestyles="--", transform=proj, zorder=1)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m"), facecolor="#9e9e9e", edgecolor="black", linewidth=0.8, zorder=2)

    # 4. Plot station data points
    sc = ax.scatter(
        surf_df["lon"], surf_df["lat"], s=100, c=surf_df[var_key],
        cmap=cmap, vmin=vmin or surf_df[var_key].min(), vmax=vmax or surf_df[var_key].max(),
        edgecolor="k", linewidth=0.4, alpha=0.9, transform=proj, zorder=3
    )

    # 5. Annotate station labels
    for lon, lat, label in zip(surf_df["lon"], surf_df["lat"], surf_df["station"]):
        ax.annotate(
            label, xy=(lon, lat), xytext=(0, 6), textcoords='offset points',
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="black",
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")]
        )

    # 6. Cosmetics (grid, scale bar, color bar, title)
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3, alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}

    scalebar = AnchoredSizeBar(ax.transData, 20 / 111, "20 km", "lower left", pad=0.4, color="k", frameon=False, size_vertical=(20/111)/20, fontproperties=fm.FontProperties(size=8))
    ax.add_artist(scalebar)

    plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.045, shrink=0.9, label=cbar_label)
    plt.title(title, fontweight="bold")
    plt.tight_layout()

    # 7. Save and show
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"✔ Map saved to {fname.name}")
    plt.show()

# ─────────────────────────
# 5. Generate Depth Profiles
# ─────────────────────────
print("--- Generating depth profile plots ---")
# Global style knobs for profiles
sns.set_theme(style="ticks", context="notebook")
profile_kws = dict(palette="tab10", markers=True, marker="o", linestyle="--", linewidth=1.8)
dcm_kws = dict(marker="o", s=150, color="green", edgecolor="black", linewidth=0.8, zorder=5, legend=False)

n_cols, n_rows = len(letters), len(nums)
for var, xlab in variables.items():
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 5.0 * n_rows), sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, station in zip(axes, stations):
        # Mean profile
        sns.lineplot(data=summary.query("station == @station & stat == 'mean'"), x=var, y="depth_m_neg", hue="cruise", hue_order=cruises, sort=False, ax=ax, **profile_kws)
        # Error bars
        for cr in cruises:
            sub_m = summary.query("station == @station & cruise == @cr & stat == 'mean'")
            sub_s = summary.query("station == @station & cruise == @cr & stat == 'std'")[var].values
            if sub_s.size:
                ax.errorbar(sub_m[var], sub_m["depth_m_neg"], xerr=sub_s, fmt="none", ecolor="gray", alpha=0.35, capsize=3)
        # DCM marker
        dcm = summary.query("station == @station & stat == 'mean' & depth_desc == 'dcm'")
        if not dcm.empty:
            sns.scatterplot(data=dcm, x=var, y="depth_m_neg", ax=ax, **dcm_kws)
        # Cosmetics
        ax.set_title(station)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Depth (m)" if station.startswith("S") else "")
        ax.set_yticks([-t for t in [0, 25, 50, 75]])
        ax.set_yticklabels([0, 25, 50, 75])
        sns.despine(ax=ax)

    for ax in axes[len(stations):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Cruise", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(f"{xlab} depth profiles – S/J/P/A grid", y=1.03)
    plt.tight_layout()

    fname = f"depth_profiles_{var}.png"
    fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight")
    print(f"✔ Profile plot saved: {fname}")
    plt.show()

# ─────────────────────────
# 6. Generate Text Summary
# ─────────────────────────
depth_labels = sorted(summary["depth_desc"].unique(),
                      key=lambda x: summary.query("depth_desc == @x")["depth_m"].mean())

bullet_list = [
    bullet(k, label, df, depth_labels)
    for k, label in variables.items()
]

print("\n\n--- Depth-profile inference bullets ---\n")
print("• Depth remains the primary driver of carbonate chemistry; surface waters differ markedly from deeper layers across all stations.")
for b in [b for b in bullet_list if b is not None]:
    print(b)


# ─────────────────────────
# 7. Generate Station Maps
# ─────────────────────────
print("\n--- Generating station-map plots ---")

# 7-a  Bathymetry subset cache
subset_nc = project_root / "ghana_bathy.nc"

if subset_nc.exists():
    bathy = xr.open_dataset(subset_nc).z
else:
    print(f"Subsetting ETOPO data from {DATA_BATHY} …")
    etopo = xr.open_dataset(DATA_BATHY)  # engine='netcdf4' is fine

    # choose correct variable name automatically
    var = "bedrock_topography" if "bedrock_topography" in etopo.data_vars else "z"

    pad = 0.3
    lon0, lon1 = df["lon"].min() - pad, df["lon"].max() + pad
    lat0, lat1 = df["lat"].min() - pad, df["lat"].max() + pad

    bathy = etopo[var].sel(lon=slice(lon0, lon1), lat=slice(lat0, lat1))
    bathy.to_netcdf(subset_nc)
    print(f"✔ Bathymetry subset saved to {subset_nc}")

# 7-b  Surface layer only
surf = (
    df.query("depth_m <= 5")
      .groupby("station", as_index=False)
      .first()                       # keeps lon/lat & carbonate vars
)

# 7-c  Variable metadata (only those present in df)
candidate_maps = {
    "pCO2"     : dict(title="Surface pCO₂",        label="pCO₂ (µatm)",        vmin=350, vmax=900),
    "Omega_ar" : dict(title="Surface Ωₐᵣ",         label="Ωₐᵣ",               vmin=0.5, vmax=3),
    "Omega_ca" : dict(title="Surface ΩCa",         label="ΩCa"),
    "HCO3"     : dict(title="Surface HCO₃⁻",       label="HCO₃⁻ (µmol kg⁻¹)"),
    "CO3"      : dict(title="Surface CO₃²⁻",       label="CO₃²⁻ (µmol kg⁻¹)"),
    "ph"       : dict(title="Surface pH",           label="pH"),
    "ta"       : dict(title="Surface TA",           label="TA (µmol kg⁻¹)"),
}

map_vars = {k: v for k, v in candidate_maps.items() if k in surf.columns}
if not map_vars:
    print("⚠ No matching carbonate columns found for maps.")
else:
    for key, meta in map_vars.items():
        fname = FIG_DIR / f"map_surface_{key}.png"
        make_station_map(
            surf_df   = surf,
            bathy_ds  = bathy,
            var_key   = key,
            title     = meta["title"],
            cbar_label= meta["label"],
            fname     = fname,
            vmin      = meta.get("vmin"),
            vmax      = meta.get("vmax"),
            cmap      = "coolwarm"
        )
