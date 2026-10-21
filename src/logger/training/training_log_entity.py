from src.logger.log_entity import log_entity
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class training_log_entity(log_entity):

    trainer_global_step : Optional[str] = None
    completion : Optional[str] = None
    token_count : Optional[int] = None
    final_answer : Optional[str] = None
    compared_final_answer : Optional[str] = None

    timeseries: Optional[np.ndarray] = None

    accuracy : Optional[bool] = None
    accuracy_reward : Optional[float] = 0.0

    synergy : Optional[float] = 0.0
    redundancy : Optional[float] = 0.0
    uniqueness : Optional[float] = 0.0
    mixed : Optional[float] = 0.0

    synergy_reward : Optional[float] = 0.0
    redundancy_reward : Optional[float] = 0.0
    synergy_redundancy_reward : Optional[float] = 0.0

    def validate(self): 
        super().validate()



