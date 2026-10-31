from src.calculator.WMSCalculator import WMSCalculator, WMSResult
from src.training.grpo_trainer import grpo_trainer
from src.logger.training.WMS_training_log_entity import WMS_training_log_entity
from src.utils.enums_class import training_type_enum, phyid_layer_type_enum, phyid_threashold_type_enum, phyid_reward_type_enum
from src.training.training_config import training_config
import torch
import numpy as np
import gc
import logging
import traceback


class WMS_grpo_trainer(grpo_trainer): 

    def __init__(self, config: training_config) -> None:
        super().__init__(config)
        self.calculator = WMSCalculator(T = 16, show_progress = False)

    def get_reward_funcs(self):
        if training_type_enum.ACCURACY_REWARD == self.config.training_type: 
            return [self.accuracy_reward]

        if training_type_enum.INFORMATION_DECOMPOSITION  == self.config.training_type: 
            return [self.calculate_information_decomposition_reward]

        if training_type_enum.INFORMATION_DECOMPOSITION_ACCURACY_REWARD  == self.config.training_type: 
            return [self.accuracy_reward, self.calculate_information_decomposition_reward]

        return []

    @torch.inference_mode()
    def calculate_information_decomposition_reward(self, completions, target=None, tokenizer=None, **kwargs):
        split_list = kwargs.get("split")     
        sample_ids = kwargs.get("sample_id") 
        problem_ids = kwargs.get("problem_id", None)
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        questions = kwargs.get("question")     
        trainer_state = kwargs.get("trainer_state", None)
        trainer_global_step = trainer_state.global_step
        
        log_list: list[WMS_training_log_entity] = super().get_log_list(trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target)
    
        rewards = []
        for log in log_list:
            if log.completion is None: 
                rewards.append(0.0)
                continue

            log = self.generate_binary_timeseries(log)
            if log.timeseries is None:
                rewards.append(0.0)
                continue
            
            reward: float = 0.0
            try:
                result: WMSResult = self.calculator.calculate(log.timeseries)
                log.joint_mi = result.joint_mi
                log.individual_mi_sum = result.individual_mi_sum
                log.wms = result.wms
                log.synergy_redundancy_reward = result.reward
                reward = log.synergy_redundancy_reward
                
            except Exception as e:
                logging.exception("An exception occurred [calculate_phyid_reward]")                        
                traceback.print_exc()                        
                print(f"[Exception] calculate_phyid_reward: {e}")
                
            rewards.append(reward)
        
        self.get_logger().write_to_log_file()
        return rewards

    @torch.inference_mode()
    def generate_binary_timeseries(self, log: WMS_training_log_entity) -> WMS_training_log_entity:
        if log.prompt is None or log.completion is None or len(log.completion) == 0:
            return log
        
        trainer = self.get_trainer()
        model = trainer.model
        tokenizer = trainer.processing_class
        
        prompt_embedding = None
        completion_embedding = None
        attention_score = None
        attention_weight = None
        try:
            refine_prompt = self.clean_prompt_for_phi(log.prompt)
            prompt_embedding = super().extract_representation(refine_prompt, model, tokenizer, self.config.layer_type)
            completion_embedding = super().extract_representation(log.completion, model, tokenizer, self.config.layer_type)
            attention_score = self.attention_score(prompt_embedding, completion_embedding)
            attention_weight = self.attention_weight(attention_score)
            time_series = np.concatenate(tuple(np.matmul(attention_weight, prompt_embedding)), axis=1,)
            
            if phyid_threashold_type_enum.AVERAGE == self.config.threashold_type: 
                avgs = np.mean(time_series, axis=0)
            elif phyid_threashold_type_enum.MEDIAN == self.config.threashold_type: 
                avgs = np.median(time_series, axis=0)
            
            time_series_copy = np.copy(time_series)
            for i in range(len(avgs)):
                time_series[np.where(time_series_copy[:, i] >= avgs[i]), i] = 1
                time_series[np.where(time_series_copy[:, i] < avgs[i]), i] = 0

            log.timeseries = time_series 
            log.completion_shape = time_series.shape
        except Exception as e:
            logging.exception("An exception occurred [generate_binary_timeseries]")                        
            traceback.print_exc()                        
            print(f"[Exception] generate_binary_timeseries: {e}")
            
        finally:
            del prompt_embedding, completion_embedding, attention_score, attention_weight
            gc.collect()

        return log
        



    