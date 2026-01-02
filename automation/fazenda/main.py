

import os
import sys
import logging
from pathlib import Path
# Add parent directory to sys.path to allow 'from core.xxx' imports when run as a script
if __name__ == "__main__" and __package__ is None:
    file = Path(__file__).resolve()
    root = file.parents[2]
    sys.path.append(str(root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from automation.fazenda.config import FazendaConfig
from automation.fazenda.report import run_report_logic
from core.s3 import S3AssetManager
from core.build_report import generar_reporte_okuo


def main() -> None:
    """
    Main execution loop for Fazenda automation using Pydantic defaults.
    """
    config = FazendaConfig()

    bucket = os.getenv("S3_BUCKET")
    s3 = S3AssetManager(notebook_name=config.notebook_name, bucket=bucket)

    logger.info(f"Starting Fazenda automation for context: {config.notebook_name}")
    config_dict = config.model_dump() 

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
    if not config.skip_report:
        logger.info("Generating final HTML report via core builder...")
        yaml_filename = os.path.basename(yaml_path)
        try:
            generar_reporte_okuo(yaml_filename)
            logger.info("Report generation complete.")
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            sys.exit(1)
    else:
        logger.info("Skipping HTML report generation as requested via config.")

if __name__ == "__main__":
    main()