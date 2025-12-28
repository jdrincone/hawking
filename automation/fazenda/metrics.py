"""
Metric calculations and statistical summaries for Fazenda.

This module provides functions to aggregate raw production data into business-level 
metrics, filter statistical outliers, and format comparative summary tables.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Any

def compute_sackoff(df: pd.DataFrame, group_cols: List[str] = []) -> pd.DataFrame:
    """
    Computes Sack-Off and production metrics grouped by specified columns.
    Always includes 'Tiene Adiflow' in the final grouping.
    """
    final_groups = group_cols + ["Tiene Adiflow"]
    
    # Define aggregation map to avoid repetition
    agg_map = {
        'op': 'nunique',
        'panificadas': 'sum',
        'Producción (Ton)': 'sum',
        'Anulación (Ton)': 'sum',
        'temp1_acond_c': 'mean',
        'pdi': 'mean',
        'finos': 'mean',
        'dureza': 'mean',
        'pdi_agro': 'mean',
        'dureza_agro': 'mean',
        'finos_agro': 'mean'
    }
    
    # Ensure columns exist before aggregating
    actual_agg = {k: v for k, v in agg_map.items() if k in df.columns}
    
    res = df.groupby(final_groups).agg(actual_agg).reset_index()
    
    # Rename for clarity
    rename_map = {
        'op': 'ops',
        'panificadas': 'production',
        'Producción (Ton)': 'reales',
        'Anulación (Ton)': 'anulation'
    }
    res = res.rename(columns=rename_map)
    
    # Calculate derived metrics
    res["diferencia"] = res["production"] - res["reales"] - res["anulation"]
    res["sackoff"] = res.apply(lambda x: (x["diferencia"] / x["reales"] * 100) if x["reales"] > 0 else 0, axis=1)
    
    return res

def filter_outliers(
    df: pd.DataFrame, 
    q_low: float = 0.01, 
    q_high: float = 0.99, 
    exclude_ops: List[str] = []
) -> pd.DataFrame:
    """
    Filters data points based on sack_off quantiles and manual exclusions.
    """
    if df.empty:
        return df
        
    low_val = df["sackoff_op"].quantile(q_low)
    high_val = df["sackoff_op"].quantile(q_high)
    
    mask = (df["sackoff_op"] >= low_val) & (df["sackoff_op"] <= high_val)
    
    if exclude_ops:
        mask &= ~df["op"].isin(exclude_ops)
        
    return df[mask].copy()

def norm_adiflow(x: Any) -> str:
    """Normalizes the 'Adiflow' labels for consistency."""
    s = str(x).lower()
    if "con" in s:
        return "Con Adiflow"
    return "Sin Adiflow"

def build_summary_comparison(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a high-level summary table comparing 'Con Adiflow' vs 'Sin Adiflow'.
    """
    if df_summary.empty:
        return pd.DataFrame()

    df = df_summary.copy()
    df["Tiene Adiflow"] = df["Tiene Adiflow"].apply(norm_adiflow)
    
    # Ensure we have both groups for comparison if possible
    pivot = df.pivot_table(
        index=None,
        columns="Tiene Adiflow",
        values=["sackoff", "reales", "diferencia", "temp1_acond_c", "pdi", "finos", "dureza"],
        aggfunc="first"
    )
    
    # Safe extraction of values
    def get_val(metric, group):
        try:
            return pivot.loc[metric, group]
        except:
            return np.nan

    ton_con = get_val("reales", "Con Adiflow")
    ton_sin = get_val("reales", "Sin Adiflow")
    sack_con = get_val("sackoff", "Con Adiflow")
    sack_sin = get_val("sackoff", "Sin Adiflow")
    
    diff_sack = sack_sin - sack_con if not (np.isnan(sack_sin) or np.isnan(sack_con)) else 0
    ton_recuperadas = (diff_sack * ton_con) / 100 if not np.isnan(ton_con) else 0
    
    res = pd.DataFrame([{
        "Toneladas Producidas Con Adiflow": round(ton_con, 2) if not np.isnan(ton_con) else 0,
        "Toneladas Producidas Sin Adiflow": round(ton_sin, 2) if not np.isnan(ton_sin) else 0,
        "Sackoff Con Adiflow (%)": round(sack_con, 2) if not np.isnan(sack_con) else 0,
        "Sackoff Sin Adiflow (%)": round(sack_sin, 2) if not np.isnan(sack_sin) else 0,
        "Diferencia Sackoff": round(diff_sack, 2),
        "Toneladas Recuperadas": round(ton_recuperadas, 2),
        "Temp Con": get_val("temp1_acond_c", "Con Adiflow"),
        "Temp Sin": get_val("temp1_acond_c", "Sin Adiflow"),
        "PDI Con": get_val("pdi", "Con Adiflow"),
        "PDI Sin": get_val("pdi", "Sin Adiflow"),
        "Finos Con": get_val("finos", "Con Adiflow"),
        "Finos Sin": get_val("finos", "Sin Adiflow"),
    }])
    
    return res

def build_pivoted_summary_table(
    df: pd.DataFrame, 
    index_col: str, 
    index_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Builds a pivoted comparison table (usually monthly or by diet).
    """
    if df.empty:
        return pd.DataFrame()

    data = df.copy()
    data["Tiene Adiflow"] = data["Tiene Adiflow"].apply(norm_adiflow)
    
    pivot = data.pivot_table(
        index=index_col,
        columns="Tiene Adiflow",
        values=["sackoff", "reales"],
        aggfunc="first"
    )
    
    # Flatten columns: ('sackoff', 'Con Adiflow') -> 'sackoff_Con Adiflow'
    pivot.columns = [f"{c[0]}_{c[1]}" for c in pivot.columns]
    
    # Fill missing groups
    for c in ["sackoff_Sin Adiflow", "sackoff_Con Adiflow", "reales_Con Adiflow", "reales_Sin Adiflow"]:
        if c not in pivot.columns:
            pivot[c] = np.nan
            
    pivot["Diferencia Sackoff"] = pivot["sackoff_Sin Adiflow"] - pivot["sackoff_Con Adiflow"]
    pivot["Toneladas Recuperadas"] = (pivot["Diferencia Sackoff"] * pivot["reales_Con Adiflow"]) / 100
    
    res = pivot.reset_index()
    
    # Final ordering if provided
    if index_order:
        res[index_col] = pd.Categorical(res[index_col].astype(str), categories=index_order, ordered=True)
        res = res.sort_values(index_col)
        
    return res.round(2)
