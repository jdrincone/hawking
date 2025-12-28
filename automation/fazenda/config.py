from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class FazendaConfig(BaseModel):
    notebook_name: str = "fazenda_efecto_adiflow_en_sackoff"
    q_low: float = Field(0.01, description="Lower quantile for outlier filtering")
    q_high: float = Field(0.99, description="Upper quantile for outlier filtering")
    exclude_ops: List[str] = ["21944", "21864"]
    descripcion: str = "Automatización del reporte de Fazenda para el análisis del efecto de Adiflow en el sackoff."
    cut_date_ensayo: str = "2025-09-01"
    
    # S3 Paths
    path_base: str = "raw/fazenda/"
    sap_file: str = "SACK OFF FAZENDA.xlsx"
    sap_sheet: str = "SAP"
    cap_sheet: str = "CAP"
    quality_file: str = "Base Fazenda.xlsx"
    quality_agro_sheet: str = "CALIDAD AGROINDUSTRIA"
    quality_control_sheet: str = "3_control_prod_peletizado"
    
    # SVG Template Path (S3)
    svg_template_path: str = f"s3://galileo-c4e9a2f1/svg_template/fazenda_sackoff.svg"
