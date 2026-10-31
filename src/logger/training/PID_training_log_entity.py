from src.logger.training.training_log_entity import training_log_entity
from dataclasses import dataclass
from typing import Optional

@dataclass
class PID_training_log_entity(training_log_entity):

    synergy : Optional[float] = 0.0
    redundancy : Optional[float] = 0.0
    uniqueness : Optional[float] = 0.0
    mixed : Optional[float] = 0.0

    synergy_reward : Optional[float] = 0.0
    redundancy_reward : Optional[float] = 0.0
    synergy_redundancy_reward : Optional[float] = 0.0

    def validate(self): 
        super().validate()



