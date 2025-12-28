"""
Modular Data loading and cleaning for El Dorado automation using SOLID principles.
"""

import sys
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from core.s3 import S3AssetManager
from automation.el_dorado.config import ElDoradoConfig

class DataLoader(ABC):
    @abstractmethod
    def load(self, s3: S3AssetManager, config: ElDoradoConfig) -> pd.DataFrame:
        pass

class MicrobiologyLoader(DataLoader):
    def load(self, s3: S3AssetManager, config: ElDoradoConfig) -> pd.DataFrame:
        df = s3.read_csv(config.microbiologia_file)
        df["date"] = pd.to_datetime(df["date"])
        
        # Consistent month labeling
        df["month_period"] = df["date"].dt.to_period("M")
        m = df["month_period"].dt.to_timestamp()
        df["month"] = m.dt.strftime("%b-%Y")
        
        return df

class EvisceradoLoader(DataLoader):
    def load(self, s3: S3AssetManager, config: ElDoradoConfig) -> pd.DataFrame:
        df = s3.read_csv(config.evicerado_file)
        df["fecha"] = pd.to_datetime(df["fecha"])
        
        # Consistent month labeling
        df["month_period"] = df["fecha"].dt.to_period("M")
        m = df["month_period"].dt.to_timestamp()
        df["month"] = m.dt.strftime("%b-%Y")
        
        return df

class ElDoradoDataProcessor:
    def __init__(self, s3: S3AssetManager, config: ElDoradoConfig):
        self.s3 = s3
        self.config = config
        self.micro_loader = MicrobiologyLoader()
        self.evi_loader = EvisceradoLoader()

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_micro = self.micro_loader.load(self.s3, self.config)
        df_evi = self.evi_loader.load(self.s3, self.config)
        
        # Determine categories for ordering based on all data
        m_combined = pd.concat([
            df_micro["month_period"].dt.to_timestamp(), 
            df_evi["month_period"].dt.to_timestamp()
        ])
        cats = pd.date_range(m_combined.min(), m_combined.max(), freq="MS").strftime("%b-%Y").tolist()
        
        # Apply categorical ordering
        df_micro["month"] = pd.Categorical(df_micro["month"], categories=cats, ordered=True)
        df_evi["month"] = pd.Categorical(df_evi["month"], categories=cats, ordered=True)
        
        return df_micro, df_evi, cats

def load_all_data(s3: S3AssetManager, config: Optional[ElDoradoConfig] = None) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    if config is None:
        config = ElDoradoConfig()
    processor = ElDoradoDataProcessor(s3, config)
    return processor.get_data()
