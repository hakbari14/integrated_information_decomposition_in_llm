from src.logger.training.training_logger import training_logger
from typing import cast
from src.logger.training.WMS_training_log_entity import WMS_training_log_entity

class WMS_training_logger(training_logger): 

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)

    def convert_buffer(self): 
        list = []
        for log in self.buffer:
            log = cast(WMS_training_log_entity, log)
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
                'Joint_Mutual_Information': log.joint_mi,
                'SUM_Individual_Mutual_Information': log.individual_mi_sum,
                'WMS': log.wms,
                'Synergy_Redundancy_Reward': log.synergy_redundancy_reward,
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
                'Joint_Mutual_Information',
                'SUM_Individual_Mutual_Information',
                'WMS',
                'Synergy_Redundancy_Reward',
                ]

        
