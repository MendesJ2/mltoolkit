from dataclasses import dataclass


@dataclass
class Feature:

    name: str

    dtype: str

    role: str

    variable_type: str

    n_unique: int

    missing_pct: float

    is_constant: bool

    is_quasi_constant: bool
