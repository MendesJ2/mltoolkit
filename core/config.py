from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:

    project_name: str

    target: str

    date_column: str | None = None

    source_column: str | None = None

    id_columns: list[str] = field(default_factory=list)

    feature_columns: list[str] | None = None
    
    ignore_columns: list[str] = field(default_factory=list)
    
    variable_types: dict[str, str] = field(
        default_factory=dict
    )

    random_state: int = 42

    test_size: float = 0.20

    output_folder: Path = Path("runs")
