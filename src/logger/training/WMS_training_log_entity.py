from src.logger.training.training_log_entity import training_log_entity
from dataclasses import dataclass
from typing import Optional

@dataclass
class WMS_training_log_entity(training_log_entity):

    joint_mi : Optional[float] = 0.0
    individual_mi_sum : Optional[float] = 0.0
    wms : Optional[float] = 0.0
    synergy_redundancy_reward : Optional[float] = 0.0

    def validate(self): 
        super().validate()



