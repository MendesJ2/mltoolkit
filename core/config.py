@dataclass
class Config:

    project_name: str

    target: str

    date_column: str

    id_columns: list[str] = field(default_factory=list)

    random_state: int = 42

    output_folder: Path = Path("runs")
