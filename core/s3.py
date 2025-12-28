"""
S3 Asset Management and AWS Integration.

This module provides a unified interface for interacting with AWS S3, including
session management (via boto3), path validation, and high-level utilities for 
reading/writing files and dataframes using awswrangler.
"""

import os
import io
import logging
import boto3
import awswrangler as wr
import pandas as pd
from typing import Optional, Any, Union
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Configure logging for S3 operations
logger = logging.getLogger(__name__)


def get_aws_session(region_name: str = "us-east-1") -> boto3.Session:
    """
    Creates a boto3 session using credentials from environment variables.

    Args:
        region_name (str, optional): AWS region to connect to (default: "us-east-1").

    Returns:
        boto3.Session: An authenticated AWS session.
    """
    return boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name
    )


def is_s3_path(path: str) -> bool:
    """
    Checks if a given string is a valid S3 URI.

    Args:
        path (str): The path/URI to check.

    Returns:
        bool: True if it starts with 's3://', False otherwise.
    """
    return str(path).startswith("s3://")


def download_from_s3(s3_path: str, as_text: bool = False) -> Union[bytes, str]:
    """
    Downloads a file from S3 and returns its content.

    Args:
        s3_path (str): The full S3 URI (e.g., "s3://bucket/key.txt").
        as_text (bool, optional): If True, decodes content as UTF-8 string. 
                                  Otherwise returns raw bytes.

    Returns:
        Union[bytes, str]: File contents.
    """
    session = get_aws_session()
    s3_client = session.client('s3')

    # Parse bucket and key from URI
    path_parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = path_parts[0]
    key = path_parts[1]

    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read()

    if as_text:
        return content.decode('utf-8')
    return content


class S3AssetManager:
    """
    Unified manager for S3 assets (Read/Write).

    Handles path construction, session persistence, and high-level operations like
    uploading local files, saving DataFrames, and storing Plotly figures or SVGs.
    """

    def __init__(
        self, 
        notebook_name: Optional[str] = None, 
        bucket: str = os.getenv("S3_BUCKET", ""), 
        region_name: str = "us-east-1"
    ):
        """
        Initializes the S3 Manager.

        Args:
            notebook_name (str, optional): Subfolder context (e.g., 'el_dorado'). 
                                           Used to prefix artifact paths.
            bucket (str): The S3 bucket name. Defaults to S3_BUCKET environment variable.
            region_name (str): AWS region.
        """
        self.bucket_name = bucket
        self.region_name = region_name
        self.notebook_name = notebook_name
        self.session = get_aws_session(region_name)
        self.s3_client = self.session.client('s3')

        # Base structure for organized storage
        self.s3_base_uri = f"s3://{self.bucket_name}"
        if self.notebook_name:
            self.data_path_uri = f"{self.s3_base_uri}/data/{self.notebook_name}"
            self.image_key_prefix = f"images/{self.notebook_name}"
        else:
            self.data_path_uri = f"{self.s3_base_uri}/data"
            self.image_key_prefix = "images"

    def read_excel(self, file: str, sheet_name: Optional[Union[str, int]] = None, **kwargs: Any) -> pd.DataFrame:
        """
        Reads an Excel file from S3 using awswrangler.

        Args:
            file (str): Key or full URI of the file.
            sheet_name (Optional[Union[str, int]]): Specific sheet to read.
            **kwargs: Additional arguments for wr.s3.read_excel.

        Returns:
            pd.DataFrame: Loaded data.
        """
        path = file if is_s3_path(file) else f"{self.s3_base_uri}/{file}"
        logger.info(f"Reading Excel from: {path}")
        return wr.s3.read_excel(path=path, sheet_name=sheet_name, boto3_session=self.session, **kwargs)

    def read_csv(self, file: str, **kwargs: Any) -> pd.DataFrame:
        """
        Reads a CSV file from S3 using awswrangler.

        Args:
            file (str): Key or full URI of the file.
            **kwargs: Additional arguments for wr.s3.read_csv.

        Returns:
            pd.DataFrame: Loaded data.
        """
        path = file if is_s3_path(file) else f"{self.s3_base_uri}/{file}"
        logger.info(f"Reading CSV from: {path}")
        return wr.s3.read_csv(path=path, boto3_session=self.session, **kwargs)
    
    def upload_file(self, local_path: str, s3_key: Optional[str] = None) -> None:
        """
        Uploads a local file to S3.

        Args:
            local_path (str): Filesystem path to the file.
            s3_key (Optional[str], optional): Destination key. If None, infers from 
                                              local_path and places it in the data folder.
        """
        if not s3_key:
             # Default to data path if no key specified
             filename = os.path.basename(local_path)
             prefix = self.data_path_uri.replace(f"s3://{self.bucket_name}/", "")
             s3_key = f"{prefix}/{filename}"

        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"File successfully uploaded to s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            logger.error(f"Error uploading file to S3: {e}")

    def save_dataframe(self, df: pd.DataFrame, filename: str, **kwargs: Any) -> None:
        """
        Saves a pandas DataFrame as a CSV directly to S3.

        Args:
            df (pd.DataFrame): Data to save.
            filename (str): Name for the CSV file.
            **kwargs: Additional arguments for df.to_csv.
        """
        full_path = f"{self.data_path_uri}/{filename}"
        logger.info(f"Saving DataFrame to CSV: {full_path}")
        # Note: df.to_csv supports S3 paths natively if credentials are set
        df.to_csv(full_path, index=False, **kwargs)

    def save_plot_static(self, fig: Any, filename: str, format: str = 'png', dpi: int = 300) -> None:
        """
        Saves a Matplotlib figure as a static image to S3.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            filename (str): Output filename.
            format (str): Image format ('png', 'jpg', etc.).
            dpi (int): Resolution.
        """
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format=format, bbox_inches='tight', dpi=dpi)
        img_buffer.seek(0)
        s3_key = f"{self.image_key_prefix}/{filename}"
        logger.info(f"Saving static plot to: s3://{self.bucket_name}/{s3_key}")
        self.s3_client.put_object(
            Bucket=self.bucket_name, 
            Key=s3_key, 
            Body=img_buffer, 
            ContentType=f'image/{format}'
        )

    def save_plotly_html(self, fig: Any, filename: str) -> None:
        """
        Saves a Plotly figure as an interactive HTML file to S3.

        Args:
            fig (plotly.graph_objects.Figure): The figure to save.
            filename (str): Output filename (e.g., "chart.html").
        """
        html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
        s3_key = f"{self.image_key_prefix}/{filename}"
        logger.info(f"Saving Plotly HTML to: s3://{self.bucket_name}/{s3_key}")
        self.s3_client.put_object(
            Bucket=self.bucket_name, 
            Key=s3_key, 
            Body=html_content.encode('utf-8'),
            ContentType='text/html'
        )

    def save_svg_content(self, svg_content: str, filename: str) -> str:
        """
        Saves a raw SVG string directly to S3.

        Args:
            svg_content (str): The SVG XML content.
            filename (str): Output filename.

        Returns:
            str: The full S3 URI of the saved file.
        """
        s3_key = f"{self.image_key_prefix}/{filename}"
        logger.info(f"Saving SVG content to: s3://{self.bucket_name}/{s3_key}")
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=svg_content.encode('utf-8'),
            ContentType='image/svg+xml'
        )
        return f"s3://{self.bucket_name}/{s3_key}"
