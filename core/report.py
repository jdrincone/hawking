"""
Utilities for report content processing and asset embedding.

This module provides functions to load YAML configurations, build asset paths,
embed images (local or S3) as Base64/DataURIs, convert DataFrames to HTML tables,
and orchestrate the processing of report sections for the rendering engine.
"""

import base64
import urllib.parse
import pandas as pd
import yaml
import os
import logging
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

from core.s3 import is_s3_path, download_from_s3, get_aws_session, wr

# Initialize logger for the report module
logger = logging.getLogger(__name__)


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file from the 'yamls' directory.

    Args:
        yaml_path (str): The filename or relative path to the YAML file (e.g., "base.yaml").

    Returns:
        Dict[str, Any]: The parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist in the 'yamls' folder.
        yaml.YAMLError: If the YAML content is malformed.
    """
    path = os.path.join("yamls", yaml_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_artifact_path(base_name: str, artifact_type: str, filename: str) -> str:
    """
    Constructs the full path (S3 or local) for a report artifact.

    The path is determined by the presence of the 'S3_BUCKET' environment variable.

    Args:
        base_name (str): The base name context (e.g., the notebook or report name).
        artifact_type (str): The subfolder category (e.g., 'images' or 'data').
        filename (str): The name of the specific file.

    Returns:
        str: A full S3 URI (s3://...) or a local filesystem path.
    """
    s3_bucket = os.getenv("S3_BUCKET")
    if s3_bucket:
        return f"s3://{s3_bucket}/{artifact_type}/{base_name}/{filename}"
    return os.path.join(artifact_type, base_name, filename)


def embed_image(image_path: str, mime_type: str = "png") -> str:
    """
    Reads an image (from S3 or local) and converts it to a Base64/DataURI string.

    This allows the final HTML report to be portable and standalone.

    Args:
        image_path (str): Full path or S3 URI to the image.
        mime_type (str, optional): The MIME type (e.g., 'png', 'jpg', 'svg').

    Returns:
        str: A formatted DataURI string (e.g., "data:image/png;base64,...").
             Returns an empty string if the image cannot be read.
    """
    try:
        if is_s3_path(image_path):
            if mime_type.lower() == "svg":
                svg_data = download_from_s3(image_path, as_text=True)
                return f"data:image/svg+xml;charset=utf-8,{urllib.parse.quote(svg_data)}"
            else:
                content = download_from_s3(image_path, as_text=False)
                encoded = base64.b64encode(content).decode("utf-8")
        else:
            if mime_type.lower() == "svg":
                with open(image_path, "r", encoding="utf-8") as f:
                    svg_data = f.read()
                return f"data:image/svg+xml;charset=utf-8,{urllib.parse.quote(svg_data)}"
            else:
                with open(image_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
        
        real_mime = "image/jpeg" if mime_type.lower() in ["jpg", "jpeg"] else f"image/{mime_type}"
        return f"data:{real_mime};base64,{encoded}"
    
    except Exception as e:
        logger.error(f"Error embedding image {image_path}: {e}")
        return ""


def dataframe_to_html(path_csv: str) -> str:
    """
    Loads a CSV file (from S3 or local) and converts it to a Bootstrap-styled HTML table.

    Args:
        path_csv (str): Full path or S3 URI to the CSV file.

    Returns:
        str: HTML string representing the table.
    """
    try:
        if is_s3_path(path_csv):
            session = get_aws_session()
            df = wr.s3.read_csv(path=path_csv, boto3_session=session)
        else:
            df = pd.read_csv(path_csv)
        return df.to_html(index=False, classes="table table-striped")
    except Exception as e:
        logger.error(f"Error converting dataframe to HTML {path_csv}: {e}")
        return f"<p>Error loading table: {e}</p>"


def read_plotly_html_file(file_path: str) -> str:
    """
    Reads the content of a Plotly-generated HTML snippet/file.

    Args:
        file_path (str): Full path or S3 URI to the HTML file.

    Returns:
        str: The raw HTML content string.
    """
    try:
        if is_s3_path(file_path):
            return download_from_s3(file_path, as_text=True)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.error(f"Error reading HTML file {file_path}: {e}")
        return f"<p>Error loading chart</p>"


def process_sections(secciones_raw: List[Dict[str, Any]], yaml_name: str) -> List[Dict[str, Any]]:
    """
    Iterates through raw report sections from YAML and prepares them for rendering.

    Processing includes:
    - Resolving image paths and embedding them as DataURIs.
    - Resolving CSV paths and converting them to HTML tables.
    - Resolving Plotly HTML paths and loading their contents.

    Args:
        secciones_raw (List[Dict[str, Any]]): List of section dictionaries from the YAML config.
        yaml_name (str): Name of the YAML file (used to derive the base artifact path).

    Returns:
        List[Dict[str, Any]]: A processed list of sections with rendered content.
    """
    base = os.path.splitext(yaml_name)[0]
    secciones = []
    
    for sec in secciones_raw:
        contenido_renderizado = []
        for bloque in sec["contenido"]:
            tipo = bloque["tipo"]
            src = bloque.get('src', '')
            
            try:
                if tipo == "imagen":
                    image_path = build_artifact_path(base, "images", src)
                    contenido_renderizado.append({
                        "tipo": "imagen",
                        "src": embed_image(image_path, bloque.get("mime", "png")),
                        "style": bloque.get("style", ""),
                    })
                elif tipo == "tabla":
                    csv_path = build_artifact_path(base, "data", src)
                    contenido_renderizado.append({
                        "tipo": "tabla",
                        "html": dataframe_to_html(csv_path),
                    })
                elif tipo == "plotly_html":
                    html_path = build_artifact_path(base, "images", src)
                    contenido_renderizado.append({
                        "tipo": "plotly_html",
                        "html_content": read_plotly_html_file(html_path),
                        "style": bloque.get("style", ""),
                    })
                else:
                    contenido_renderizado.append(bloque)
            except Exception as e:
                logger.error(f"Error processing block {tipo}: {e}")
                contenido_renderizado.append({"tipo": "error", "msg": str(e)})

        secciones.append({"titulo": sec["titulo"], "contenido": contenido_renderizado})
    return secciones


def ensure_dirs(notebook_name: str) -> Tuple[Path, Path]:
    """
    Ensures that the local directories for data and images exist for a given context.

    Args:
        notebook_name (str): The name of the notebook/report context.

    Returns:
        Tuple[Path, Path]: A tuple containing the (data_path, images_path) as Path objects.
    """
    root_data = Path(f"data/{notebook_name}")
    root_images = Path(f"images/{notebook_name}")
    root_data.mkdir(parents=True, exist_ok=True)
    root_images.mkdir(parents=True, exist_ok=True)
    return root_data, root_images
