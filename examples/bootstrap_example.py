from pathlib import Path
import sys


# =========================================================
# Workspace
# =========================================================

WORKSPACE_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(WORKSPACE_ROOT),
    )


# =========================================================
# Project
# =========================================================

from examples.project_template.project import (
    ProjectTemplate,
)


project = ProjectTemplate(
    logger=None
)
