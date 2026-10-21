from enum import StrEnum

class dataset_element_type_enum(StrEnum):
    TRAIN = 'train'
    EVAL = 'eval'

class phyid_log_type_enum(StrEnum):
    TRAIN_TEST = 'train_test'
    TEST = 'test'

class phyid_layer_type_enum(StrEnum):
    ALL = 'all'
    LAST = 'last'
    SOME = 'some'

class phyid_threashold_type_enum(StrEnum):
    AVERAGE = 'average'
    MEDIAN = 'median'

class training_type_enum(StrEnum):
    ACCURACY_REWARD = 'accuracy_reward'
    ACCURACY_REWARD_PHY_ID = 'accuracy_reward_phy_id'

class phyid_reward_type_enum(StrEnum):
    SYNERGY = 'synergy'
    REDUNDANCY = 'redundancy'
    SYNERGY_REDUNDANCY = 'synergy_redundancy'

class llm_pipeline_type_enum(StrEnum):
    TRAINING = 'training'
    INFERENCE = 'inference'

