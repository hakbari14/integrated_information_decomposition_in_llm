from dataclasses import dataclass
from src.utils.enums_class import training_type_enum, phyid_layer_type_enum, phyid_reward_type_enum, phyid_threashold_type_enum
from typing import Optional


@dataclass
class training_config:
    model_name: str
    
    training_type: training_type_enum 
    layer_type: phyid_layer_type_enum 
    threashold_type: phyid_threashold_type_enum 

    reduced_dimention: Optional[int] = None
    reward_type: Optional[phyid_reward_type_enum] = None
    
    def to_dict(self):
        return self.__dict__