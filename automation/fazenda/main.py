"""
CLI Entry point for the Fazenda report automation.

This script parses command-line arguments to customize the report parameters
via the FazendaConfig model and triggers the orchestration logic.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from core.s3 import S3AssetManager
from core.build_report import generar_reporte_okuo
from automation.fazenda.report import run_report_logic
from automation.fazenda.config import FazendaConfig

# Configure global logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main execution loop for Fazenda automation.
    """
    parser = argparse.ArgumentParser(description="Run Fazenda Report Automation")
    parser.add_argument(
        "--notebook_name", 
        default="fazenda_efecto_adiflow_en_sackoff", 
        help="Context name for S3 assets storage"
    )
    parser.add_argument(
        "--q_low", type=float, default=0.01, help="Lower quantile for outlier filtering"
    )
    parser.add_argument(
        "--q_high", type=float, default=0.99, help="Upper quantile for outlier filtering"
    )
    parser.add_argument(
        "--skip_report", 
        action="store_true", 
        help="If set, skips the final HTML generation step"
    )

    args = parser.parse_args()

    # 1. Initialize S3
    bucket = os.getenv("S3_BUCKET")
    s3 = S3AssetManager(notebook_name=args.notebook_name, bucket=bucket)

    # 2. Build Config Dict for Pydantic (or pass directly if orchestrator allows)
    config_dict = {
        'notebook_name': args.notebook_name,
        'q_low': args.q_low,
        'q_high': args.q_high,
    }

    # 3. Run Automation Logic
    logger.info(f"Starting Fazenda automation for context: {args.notebook_name}")
    try:
        yaml_path = run_report_logic(s3, config_dict)
        if not yaml_path:
            logger.error("Automation logic failed to return a YAML path.")
            sys.exit(1)
        logger.info(f"YAML configuration generated successfully at: {yaml_path}")
    except Exception as e:
        logger.error(f"Error during automation logic: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Generate Final HTML Report
    if not args.skip_report:
        logger.info("Generating final HTML report via core builder...")
        yaml_filename = os.path.basename(yaml_path)
        try:
            generar_reporte_okuo(yaml_filename)
            logger.info("Report generation complete.")
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
