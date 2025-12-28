from pydantic import BaseModel, Field
from typing import List, Optional

class ElDoradoConfig(BaseModel):
    notebook_name: str = "el_dorado_rendering"
    
    # S3 Paths
    microbiologia_file: str = "raw/dorado/rendering_microbiologia/microbiologia_el_dorado.csv"
    evicerado_file: str = "raw/dorado/rendering_microbiologia/evicerado_el_dorado.csv"
    
    # CDF Configuration
    cdf_columns: List[str] = ["ph", "orp", "cloro"]
    
    # Visualization Settings
    line_colors: dict = {
        "Poll Ext Visceras": "#1f4e5f",
        "ciegos": "#3d8b7a",
        "salida chiller": "#9db494",
        "prechiller": "#f2b47e",
        "Descuento (Kg)": "#ef8a82"
    }
    
    # Description for YAML
    descripcion: str = "Automatización del reporte El Dorado para el análisis de microbiología y eviscerado."
