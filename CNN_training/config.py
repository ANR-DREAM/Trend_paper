from dataclasses import dataclass

@dataclass
class Params:
    model: str
    dropout: float
    val_split: float
    lr : float
    batch_size : int
    n_epochs : int
    random_state : int

@dataclass
class Data:
    input: str
    ouput: str
    model_type: str
    model_type_predictors: str
    sensor : str
    anomaly : bool
    d1_train: str
    d2_train: str
    d1_pred: str
    d2_pred: str
    d1_test: str
    d2_test: str
    plot_date: str
    lat_min: int
    lat_max: int
    lon_min: int
    lon_max: int

@dataclass
class Out:
    training : bool
    prediction : bool
    metric : bool
    plot : bool

@dataclass
class Path:
    log : str
    data_input : str
    data_output : str
    
@dataclass
class CHLConfig:
    params: Params
    path: Path
    data: Data
    out: Out
