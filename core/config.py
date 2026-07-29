from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:

    project_name: str

    target: str

    date_column: str | None = None

    output_folder: Path = Path("runs")

    random_state: int = 42

    test_size: float = 0.20
