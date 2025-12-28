"""
Script for generating HTML reports from YAML configuration files.

This module orchestrates the process of loading configuration, embedding assets (logos, favicons),
processing report sections (images, tables, Plotly charts), and rendering the final HTML
using Jinja2 templates. It also handles uploading the generated report to S3.
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any

# Add parent directory to sys.path to allow 'from core.xxx' imports when run as a script
if __name__ == "__main__" and __package__ is None:
    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))

from jinja2 import Environment, FileSystemLoader
from core.report import embed_image, load_yaml_config, process_sections
from core.s3 import S3AssetManager

# Configure logging to provide visibility during report generation
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Absolute path to the core directory for reliable template and asset loading
core_dir = os.path.dirname(os.path.abspath(__file__))


def generar_reporte_okuo(yaml_file: str) -> None:
    """
    Generates a full HTML report based on the provided YAML configuration.

    This function performs the following steps:
    1.  Initializes the Jinja2 environment and loads the base template.
    2.  Sets up metadata such as the current date and embeds static assets (logo/favicon).
    3.  Loads the report configuration from a YAML file.
    4.  Processes each section of the report, converting data and images into embeddable formats.
    5.  Renders the HTML content and saves it to a local 'reports' directory.
    6.  Automatically opens the generated report in the system's default web browser.
    7.  Uploads the final report to the designated S3 bucket.

    Args:
        yaml_file (str): Name or path of the YAML configuration file (e.g., "production_report.yaml").
    
    Raises:
        FileNotFoundError: If the YAML file or required templates/assets are missing.
        Exception: For any errors during processing or S3 upload.
    """
    # Path to the Jinja2 templates directory
    templates_path = os.path.join(core_dir, "templates")

    # Initialize Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_path))
    template = env.get_template("v1.html")

    # Formatted current date for the report footer/header
    fecha = datetime.now().strftime("%d-%m-%Y")

    # Paths to static branding assets
    logo_path = os.path.join(core_dir, "static", "logo_ppal.jpg")
    favicon_path = os.path.join(core_dir, "static", "logo.png")

    # Embed static images as Base64/DataURIs for standalone HTML portability
    logo = embed_image(logo_path, "jpg")
    favicon = embed_image(favicon_path, "png")

    # Load and process the report configuration and its sections
    data = load_yaml_config(yaml_file)
    secciones = process_sections(data["secciones"], yaml_file)

    # Render the final HTML with all data context
    html = template.render(
        titulo=data["titulo"],
        lugar=data["lugar"],
        descripcion=data.get("descripcion", ""),
        secciones=secciones,
        fecha=fecha,
        logo=logo,
        favicon=favicon,
    )

    # Ensure output directory exists
    output_dir = "reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the output filename based on the YAML filename
    base_name = os.path.splitext(os.path.basename(yaml_file))[0]
    nombre_salida = os.path.join(output_dir, f"informe_{base_name}.html")

    # Write the rendered HTML to file
    with open(nombre_salida, "w", encoding="utf-8") as f:
        f.write(html)

    # Provide immediate feedback by opening the file
    webbrowser.open(f"file://{os.path.abspath(nombre_salida)}")
    logger.info(f"Report successfully generated locally: {nombre_salida}")

    # Synchronize the report to S3 for remote access/sharing
    s3_key = f"reports/{os.path.basename(nombre_salida)}"
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        s3_manager = S3AssetManager(bucket=bucket)
        s3_manager.upload_file(nombre_salida, s3_key)
        logger.info(f"Report uploaded to S3: s3://{bucket}/{s3_key}")
    else:
        logger.warning("S3_BUCKET environment variable not set. Skipping S3 upload.")


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the report generation script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments, specifically 'name'.
    """
    parser = argparse.ArgumentParser(
        description="Generate an HTML report from a YAML configuration file"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the YAML file for the report (e.g., base.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Execute the report generation process
    args = parse_args()
    generar_reporte_okuo(args.name)
