from dataclasses import dataclass

@dataclass
class Data:
    input: str
    d1: str
    d2: str
    lat_min: int
    lat_max: int
    lon_min: int
    lon_max: int
    plot_only : bool
    plot : bool

@dataclass
class Path:
    log : str
    path_save : str
    
@dataclass
class CHLConfig:
    path: Path
    data: Data
