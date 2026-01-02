"""
Main entry point for the Fazenda automation process.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path to allow imports from core and automation
if __name__ == "__main__" and __package__ is None:
    file = Path(__file__).resolve()
    root = file.parents[2]
    sys.path.append(str(root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from automation.fazenda.config import FazendaConfig
from automation.fazenda.report import FazendaReportOrchestrator
from core.s3 import S3AssetManager

def main() -> None:
    """
    Main execution flow for Fazenda automation.
    """
    try:
        config = FazendaConfig()
        bucket = os.getenv("S3_BUCKET")
        s3 = S3AssetManager(notebook_name=config.notebook_name, bucket=bucket)

        logger.info(f"Starting Fazenda automation: {config.notebook_name}")
        orchestrator = FazendaReportOrchestrator(s3=s3, config=config)
        orchestrator.run()

    except Exception as e:
        logger.exception(f"An error occurred during automation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()