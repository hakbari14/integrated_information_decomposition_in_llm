from src.logger.logger import logger
from typing import cast
from src.logger.training.training_log_entity import training_log_entity

class training_logger(logger): 

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)

    def get_log_list(self, trainer_global_step): 
        return list(filter(lambda x: x.trainer_global_step == trainer_global_step , self.buffer))

    def convert_buffer(self): 
        list = []
        for log in self.buffer:
            log = cast(training_log_entity, log)
            b = { 
                'ID': log.ID, 
                'Trainer_Global_Step': log.trainer_global_step, 
                'Split': log.split, 
                'Sample_ID': log.sample_ID, 
                'problem_id': log.problem_id, 
                'Question': log.question, 
                'Prompt': log.prompt, 
                'Target': log.target, 
                'Completion': log.completion, 
                'Completion_Shape': log.completion_shape, 
                'Token_Count': log.token_count, 
                'Final_Answer': log.final_answer,
                'Compared_Final_Answer': log.compared_final_answer,
                'Accuracy_Reward': log.accuracy_reward,
                'Accuracy': log.accuracy,
                'Synergy': log.synergy,
                'Synergy_Reward': log.synergy_reward,
                'Redundancy': log.redundancy,
                'Redundancy_Reward': log.redundancy_reward,
                'Synergy_Redundancy_Reward': log.synergy_redundancy_reward,
                'Uniqueness': log.uniqueness,
                'Mixed': log.mixed,
                }
            list.append(b)            
        return list

    def get_fieldnames(self): 
        return [ 
                'ID', 
                'Trainer_Global_Step', 
                'Split', 
                'Sample_ID', 
                'problem_id', 
                'Question', 
                'Prompt', 
                'Target', 
                'Completion', 
                'Completion_Shape', 
                'Token_Count', 
                'Final_Answer',
                'Compared_Final_Answer',
                'Accuracy_Reward',
                'Accuracy',
                'Synergy',
                'Synergy_Reward',
                'Redundancy',
                'Redundancy_Reward',
                'Synergy_Redundancy_Reward',
                'Uniqueness',
                'Mixed',
                ]

        
