"""
Modular Data loading and cleaning for Fazenda automation using SOLID principles.

This module provides specialized loaders and processors to handle datasets 
from S3: SAP orders, CAP weights, and Quality Control metrics.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Protocol, List, Dict, Any, Optional
from abc import ABC, abstractmethod

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from core.s3 import S3AssetManager
from automation.fazenda.config import FazendaConfig

class DataLoader(ABC):
    """Abstract base class for data loading."""
    @abstractmethod
    def load(self, s3: S3AssetManager, config: FazendaConfig) -> pd.DataFrame:
        pass

class SAPLoader(DataLoader):
    """Loads and cleans SAP production order data."""
    def load(self, s3: S3AssetManager, config: FazendaConfig) -> pd.DataFrame:
        df = s3.read_excel(f"{config.path_base}{config.sap_file}", sheet_name=config.sap_sheet)
        df = df[df["cerrada o abierta"] == 1]
        df.columns = [str(x).strip() for x in df.columns]
        df = df[df["Orden"].notnull()]
        
        rename_map = {
            'Orden': 'order',
            'Descripción': "Dieta",
            'OP CAP': "op",
            'Fecha liberacion': "date",
            'Cantidad planificada': "panificadas",
            'Cantidad entregada': "entregadas",
            "101": "code_101",
            "102": "Anulación (Ton)",
            "122": "code_122",
            "309": "code_309",
            "261": "code_261",
            "262": "code_262",
            "641": "code_641",
            "642": "code_642",
            "PRODUCCION": "Producción (Ton)",
            'DIF PRO - CONS': "diff_prod",
            'SACKOFF %': "sackoff",
            'SACK OFF PROD': "sackoff_prod",
            'cerrada o abierta': "status"
        }
        df = df.rename(columns=rename_map)
        
        # Numeric conversion and scaling (Grams to Tons)
        numeric_cols = [
            "panificadas", "entregadas", "code_101", "Anulación (Ton)",
            "code_122", "code_309", "code_261", "code_262", 
            "code_641", "code_642", "Producción (Ton)", "diff_prod"
        ]
        available_cols = [c for c in numeric_cols if c in df.columns]
        for c in available_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 1000
            
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M")
        df["op"] = pd.to_numeric(df["op"], errors="coerce")
        
        # Calculate derived metrics
        df["diff"] = df["code_101"] - df["Producción (Ton)"] - df["Anulación (Ton)"]
        den = df["Producción (Ton)"]
        df["sackoff_op"] = np.where(den > 0, df["diff"] / den * 100, 0.0)
        
        return df[list(rename_map.values()) + ["month", "diff", "sackoff_op"]]

class CAPLoader(DataLoader):
    """Loads and cleans CAP weight data."""
    def load(self, s3: S3AssetManager, config: FazendaConfig) -> pd.DataFrame:
        df = s3.read_excel(f"{config.path_base}{config.sap_file}", sheet_name=config.cap_sheet)
        df = df[df["O.P."].notnull()]
        
        rename_map = {"O.P.": "op", "Peso real": "peso_real", "Peso Agua": "peso_agua"}
        df = df.rename(columns=rename_map)
        df_dep = df[list(rename_map.values())].copy()
        
        for c in ["peso_real", "peso_agua"]:
            df_dep[c] = pd.to_numeric(df_dep[c], errors="coerce") / 1000
            
        df_dep["op"] = pd.to_numeric(df_dep["op"], errors="coerce")
        return df_dep

class QualityLoader(DataLoader):
    """Loads and aggregates Quality Control data (both Agro and Control)."""
    def load(self, s3: S3AssetManager, config: FazendaConfig) -> pd.DataFrame:
        # 1. Control Prod Peletizado
        qa_control = s3.read_excel(f"{config.path_base}{config.quality_file}", sheet_name=config.quality_control_sheet, skiprows=3)
        
        # Aggregate standard OP columns
        qa_control12 = qa_control.groupby(['OP CAP']).agg(
            product_name=('PRODUCTO', 'first'),
            temp1_acond_c=('TEMPERATURA DEL ACONDICIONADOR (°C) Pelet 1', 'mean'),
            pdi=('DURABILIDAD (%)', 'mean'),
            dureza=('DUREZA (kg/cm2)', 'mean'),
            finos=('FINOS (%)', 'mean')
        ).reset_index().rename(columns={'OP CAP': 'op'})
        
        # Aggregate alternate OP columns (- OP CAP)
        qa_control3 = qa_control.groupby(['- OP CAP']).agg(
            product_name=('- PRODUCTO', 'first'),
            temp1_acond_c=('- TEMPERATURA DEL ACONDICIONADOR (°C) Pelet 3', 'mean'),
            pdi=('- DURABILIDAD (%)', 'mean'),
            dureza=('- DUREZA (kg/cm2)', 'mean'),
            finos=('- FINOS (%)', 'mean')
        ).reset_index().rename(columns={'- OP CAP': 'op'})
        
        qa_comp = pd.concat([qa_control12, qa_control3]).drop_duplicates(subset=['op'], keep="first")
        qa_comp["op"] = pd.to_numeric(qa_comp["op"], errors='coerce')
        qa_comp = qa_comp.dropna(subset=["op"])
        
        # 2. Agroindustry Quality
        qa_agro = s3.read_excel(f"{config.path_base}{config.quality_file}", sheet_name=config.quality_agro_sheet)
        agro_numeric = ['Durabilidad pellet ', 'Dureza pellet ', 'Finos pellet']
        for cl in agro_numeric:
            qa_agro[cl] = pd.to_numeric(qa_agro[cl], errors='coerce')
            
        qa_agro_comp = qa_agro.groupby(['OP CAP']).agg(
            pdi_agro=('Durabilidad pellet ', 'mean'),
            dureza_agro=('Dureza pellet ', 'mean'),
            finos_agro=('Finos pellet', 'mean')
        ).reset_index().rename(columns={'OP CAP': 'op'})
        qa_agro_comp["op"] = pd.to_numeric(qa_agro_comp["op"], errors='coerce')
        qa_agro_comp = qa_agro_comp.dropna(subset=["op"])
        
        # Merge Quality Data
        qa_final = pd.merge(qa_comp, qa_agro_comp, on="op", how="outer")
        return qa_final

class FazendaDataProcessor:
    """Orchestrates loading, cleaning and merging of all Fazenda datasets."""
    def __init__(self, s3: S3AssetManager, config: FazendaConfig):
        self.s3 = s3
        self.config = config
        self.sap_loader = SAPLoader()
        self.cap_loader = CAPLoader()
        self.qa_loader = QualityLoader()

    def get_consolidated_data(self) -> pd.DataFrame:
        """Loads and merges all datasets into a single tidy DataFrame."""
        df_sap = self.sap_loader.load(self.s3, self.config)
        df_cap = self.cap_loader.load(self.s3, self.config)
        df_qa = self.qa_loader.load(self.s3, self.config)
        
        # Merge SAP and CAP
        df = pd.merge(df_sap, df_cap, on='op', how='left')
        
        # Determine Adiflow usage
        df["Tiene Adiflow"] = np.where(df["peso_agua"] > 0, "Con Adiflow", "Sin Adiflow")
        df = df[df["op"].notnull()]
        
        # Merge Quality
        df = pd.merge(df, df_qa, on='op', how='left')
        
        # Filter by cut date
        df["op"] = df["op"].astype(int).astype(str)
        df = df[df["date"] >= self.config.cut_date_ensayo]
        
        return df

def load_and_clean_data(s3: S3AssetManager, config: Optional[FazendaConfig] = None) -> pd.DataFrame:
    """Maintain backward compatibility while using the new modular classes."""
    if config is None:
        config = FazendaConfig()
    processor = FazendaDataProcessor(s3, config)
    return processor.get_consolidated_data()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s3 = S3AssetManager(notebook_name="fazenda_efecto_adiflow_en_sackoff")
    cfg = FazendaConfig()
    df = load_and_clean_data(s3, cfg)
    print(df.head())
    print(f"Loaded {len(df)} rows.")