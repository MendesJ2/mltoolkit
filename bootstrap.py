# bootstrap.py

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from projects.insurance.project import InsuranceProject
from projects.insurance.config import config

from mltoolkit.core.logger import create_logger


logger = create_logger()


project = InsuranceProject(
    config=config,
    logger=logger
)
