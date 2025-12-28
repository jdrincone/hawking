"""
Metric calculations and statistical summaries for El Dorado.
"""

import pandas as pd
import numpy as np
from typing import List

def compute_micro_prevalence(df_micro: pd.DataFrame) -> pd.DataFrame:
    """Calculates microbiological prevalence and other metrics per month/etapa."""
    res = df_micro.groupby(["month", "etapa", "microorganismo"]).agg(
        prev=("have_micro", "mean"),
        result=('result', 'mean'),
        ph=('ph', 'mean'),
        orp=('orp', 'mean'),
        cloro=('cloro', 'mean'),
        n_analysis=('have_micro', 'count')
    ).reset_index()
    
    res["prev"] = res["prev"] * 100
    return res

def compute_eviscerado_summary(df_evi: pd.DataFrame) -> pd.DataFrame:
    """Aggregates evisceration data and calculates total discounts."""
    res = df_evi.groupby("month").agg(
        pollos_remisionados=('pollos_remisionados', "sum"),
        buches_eviceracion=('buches_zona_eviceración', "sum"),
        intestinos_eviceracion=('intestinos_zona_eviceración', "sum"),
        gr_material_fecal=('gr_materia_fecal', "sum"),
    ).reset_index()
    
    res["kg_descuento"] = res["buches_eviceracion"] + res["intestinos_eviceracion"]
    return res

def prepare_combined_metrics(df_micro_month: pd.DataFrame, df_evi_month: pd.DataFrame, microorganism: str) -> pd.DataFrame:
    """Combines micro and eviscerado data for a specific microorganism."""
    df_micro_sub = df_micro_month[df_micro_month["microorganismo"] == microorganism].copy()
    
    # Pre-calculate kilos discount as a separate 'etapa' for plotting compatibility
    df_kilos = df_evi_month[["month", "kg_descuento"]].copy()
    df_kilos["etapa"] = "Descuento (Kg)"
    df_kilos = df_kilos.rename(columns={"kg_descuento": "prev"})
    
    return pd.concat([df_micro_sub, df_kilos], ignore_index=True)

def compute_farm_prevalence(df_micro: pd.DataFrame) -> pd.DataFrame:
    """Calculates prevalence grouped by farm, etapa, and microorganism."""
    res = df_micro.groupby(["month", "etapa", "granja", "microorganismo"]).agg(
        prev=("have_micro", "mean"),
        result=('result', 'mean'),
        ph=('ph', 'mean'),
        orp=('orp', 'mean'),
        cloro=('cloro', 'mean'),
        n_analysis=('have_micro', 'count')
    ).reset_index()
    
    res["prev"] = res["prev"] * 100
    return res

def filter_last_month_data(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Filters data only for the most recent month in the dataset."""
    if df.empty:
        return df
    last_month_period = df[date_col].dt.to_period("M").max()
    return df[df[date_col].dt.to_period("M") == last_month_period].copy()
