from __future__ import annotations
from datetime import date, datetime
from pathlib import Path
from jinja2 import Environment, BaseLoader
from core.s3 import download_from_s3


def fmt_num(x, decimals=2, thousands=".", decimal=",") -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    s = f"{float(x):,.{decimals}f}"
    return s.replace(",", "_").replace(".", decimal).replace("_", thousands)

def fmt_date(x, fmt="%d-%b") -> str:
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, datetime): x = x.date()
    if isinstance(x, date): return x.strftime(fmt)
    return str(x)

env = Environment(loader=BaseLoader(), autoescape=False)

def generate_sackoff_svg(
    template_path: str | Path,
    *,
    ton_con_adiflow, ton_sin_adiflow, mejora_pct,
    sackoff_con, sackoff_sin, recuperadas_prom,
    temp_con, temp_sin, delta_temp,
    pdi_con, pdi_sin, finos_con, finos_sin,
    fecha_ini, fecha_fin, pct_datos,
) -> str:
    """
    Renderiza el SVG y devuelve el STRING (contenido), no guarda nada en disco.
    Admite template_path local o s3://.
    """
    
    # 1. Obtener el contenido del template
    template_path_str = str(template_path)
    
    template_str = download_from_s3(template_path_str, as_text=True)
    
    # 2. Contexto de datos
    context = {
        "ton_con_adiflow": fmt_num(ton_con_adiflow, 1),
        "ton_sin_adiflow": fmt_num(ton_sin_adiflow, 2),
        "mejora_pct": fmt_num(mejora_pct, 2),
        "sackoff_con": fmt_num(sackoff_con, 2),
        "sackoff_sin": fmt_num(sackoff_sin, 2),
        "recuperadas_prom": fmt_num(recuperadas_prom, 1),
        "temp_con": fmt_num(temp_con, 2),
        "temp_sin": fmt_num(temp_sin, 2),
        "delta_temp": fmt_num(delta_temp, 1),
        "pdi_con": fmt_num(pdi_con, 2),
        "pdi_sin": fmt_num(pdi_sin, 2),
        "finos_con": fmt_num(finos_con, 2),
        "finos_sin": fmt_num(finos_sin, 2),
        "fecha_ini": fmt_date(fecha_ini, "%d-%b"),
        "fecha_fin": fmt_date(fecha_fin, "%d-%b"),
        "pct_datos": fmt_num(pct_datos, 0),
    }

    # 3. Retornar el string renderizado
    return env.from_string(template_str).render(**context)