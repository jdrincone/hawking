"""
Metric calculation functions for the Fazenda automation report.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

def norm_adiflow(x: Any) -> str:
    """
    Normalizes Adiflow labels to 'Con Adiflow' or 'Sin Adiflow'.
    """
    s = str(x).lower()
    if "con" in s and "adiflow" in s:
        return "Con Adiflow"
    if "sin" in s and "adiflow" in s:
        return "Sin Adiflow"
    if s in ["1", "si", "sí", "true"]:
        return "Con Adiflow"
    if s in ["0", "no", "false"]:
        return "Sin Adiflow"
    return s

def _calculate_impact_metrics(fila_con: pd.Series, fila_sin: pd.Series, s_col: str, p_col: str) -> Dict[str, float]:
    """
    Helper function to calculate sackoff difference and recovered tons.
    """
    sack_con = float(fila_con[s_col])
    sack_sin = float(fila_sin[s_col])
    diff_sack = sack_con - sack_sin
    
    ton_con = float(fila_con[p_col])
    ton_recuperadas = ton_con * diff_sack / 100.0
    
    return {
        "Sackoff Sin Adiflow": round(sack_sin, 2),
        "Sackoff Con Adiflow": round(sack_con, 2),
        "Diferencia Sackoff": round(diff_sack, 2),
        "Toneladas Producidas Con Adiflow": round(ton_con, 2),
        "Toneladas Recuperadas": round(ton_recuperadas, 2),
    }

def compute_sackoff(
    df: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    """
    Computes sackoff and other aggregated metrics by specified columns.
    """
    col_adiflow = "Tiene Adiflow"
    sackoff_name = "sackoff"
    group_cols = cols + [col_adiflow]

    df_group = (
        df
        .groupby(group_cols, dropna=False)
        .agg(
            diferencia=("diff", "sum"),
            reales=("peso_real", "sum"),
            production=("Producción (Ton)", 'sum'),
            anulation=("Anulación (Ton)", 'sum'),
            sackoff_mean=('sackoff_op', 'median'),
            ops=('op', 'nunique'),
            temp1_acond_c=('temp1_acond_c', 'mean'),
            pdi=('pdi', "mean"),
            pdi_agro=('pdi_agro', "mean"),
            finos=('finos', "mean"),
            finos_agro=('finos_agro', "mean"),
            dureza=('dureza', "mean"),
            dureza_agro=('dureza_agro', "mean")
        )
        .reset_index()
    )
    df_group[sackoff_name] = np.where(
        df_group["reales"] > 0,
        df_group["diferencia"] / df_group["reales"] * 100,
        np.nan
    )

    return df_group


def build_summary_table(summary_general_cut: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a summary table showing sackoff comparison and recovered tons.
    """
    df = summary_general_cut.copy()
    df["Tiene Adiflow std"] = df["Tiene Adiflow"].apply(norm_adiflow)

    fila_con = df[df["Tiene Adiflow std"] == "Con Adiflow"].iloc[0]
    fila_sin = df[df["Tiene Adiflow std"] == "Sin Adiflow"].iloc[0]

    s_col = "Sackoff Prom (%)" if "Sackoff Prom (%)" in df.columns else "Sackoff (%)"
    p_col = "Producidas (Ton)"

    resumen_dict = _calculate_impact_metrics(fila_con, fila_sin, s_col, p_col)
    return pd.DataFrame([resumen_dict])


def build_monthly_summary_table(
    df: pd.DataFrame,
    month_col: str = "month",
    adiflow_col: str = "Tiene Adiflow",
    produced_col: str = "Producidas (Ton)",
    sackoff_col: str = "Sackoff (%)",
    category_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Builds a summary table showing sackoff comparison per month or category (Dieta).
    """
    data = df.copy()
    data["Tiene Adiflow std"] = data[adiflow_col].apply(norm_adiflow)

    rows = []
    for cat, g in data.groupby(month_col):
        fila_con = g[g["Tiene Adiflow std"] == "Con Adiflow"]
        fila_sin = g[g["Tiene Adiflow std"] == "Sin Adiflow"]

        if fila_con.empty or fila_sin.empty:
            continue

        impact_data = _calculate_impact_metrics(fila_con.iloc[0], fila_sin.iloc[0], sackoff_col, produced_col)
        impact_data["Category"] = cat
        rows.append(impact_data)

    resumen = pd.DataFrame(rows)

    if category_order is not None and not resumen.empty:
        cat_type = pd.Categorical(resumen["Category"], categories=category_order, ordered=True)
        resumen = resumen.assign(Category=cat_type).sort_values("Category").reset_index(drop=True)

    # Ensure Category is the first column for better readability
    if not resumen.empty:
        cols = ['Category'] + [c for c in resumen.columns if c != 'Category']
        resumen = resumen[cols]

    return resumen


def extract_svg_params(df: pd.DataFrame, fecha_ini: str, fecha_fin: str, pct_datos: int) -> dict:
    """
    Extracts parameters from a summary DataFrame for SVG generation.
    """
    df_mapped = df.copy()
    df_mapped["Tiene Adiflow"] = df_mapped["Tiene Adiflow"].apply(norm_adiflow)
    data_dict = df_mapped.set_index('Tiene Adiflow').to_dict(orient='index')
    
    con = data_dict.get('Con Adiflow', {})
    sin = data_dict.get('Sin Adiflow', {})
    
    def get_val(d, *keys):
        for k in keys:
            if k in d: return d[k]
        return 0

    s_con = get_val(con, 'Sackoff Prom (%)', 'Sackoff (%)')
    s_sin = get_val(sin, 'Sackoff Prom (%)', 'Sackoff (%)')
    t_con = get_val(con, 'Producidas (Ton)')
    mejora = s_con - s_sin
    
    return {
        "ton_con_adiflow": t_con,
        "ton_sin_adiflow": get_val(sin, 'Producidas (Ton)'),
        "mejora_pct": mejora, 
        "sackoff_con": s_con,
        "sackoff_sin": s_sin,
        "recuperadas_prom": mejora * t_con / 100, 
        "temp_con": get_val(con, 'Temp (°C)'),
        "temp_sin": get_val(sin, 'Temp (°C)'),
        "delta_temp": get_val(con, 'Temp (°C)') - get_val(sin, 'Temp (°C)'),
        "pdi_con": get_val(con, 'Pdi (%)'),
        "pdi_sin": get_val(sin, 'Pdi (%)'),
        "finos_con": get_val(con, 'Finos (%)'),
        "finos_sin": get_val(sin, 'Finos (%)'),
        "fecha_ini": fecha_ini,
        "fecha_fin": fecha_fin,
        "pct_datos": pct_datos,
    }
