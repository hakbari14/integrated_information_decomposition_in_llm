from src.calculator.PIDCalculator import PIDCalculator, PIDResult
from src.training.grpo_trainer import grpo_trainer
from src.logger.training.PID_training_log_entity import PID_training_log_entity
from src.utils.enums_class import training_type_enum, phyid_threashold_type_enum, phyid_reward_type_enum
from src.training.training_config import training_config
from sklearn import decomposition
import torch
import numpy as np
import gc
import logging
import traceback


class PID_grpo_trainer(grpo_trainer): 

    def __init__(self, config: training_config) -> None:
        super().__init__(config)
        self.calculator = PIDCalculator()

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
        
        log_list: list[PID_training_log_entity] = super().get_log_list(trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target)
    
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
                result: PIDResult = self.calculator.compute(log.timeseries)
                log.synergy = result.synergy
                log.redundancy = result.redundancy
                log.uniqueness = result.uniqueness
                log.mixed = result.mixed
                
                if phyid_reward_type_enum.SYNERGY == self.config.reward_type: 
                    log.synergy_reward = log.synergy
                    reward = log.synergy_reward
                elif phyid_reward_type_enum.REDUNDANCY == self.config.reward_type:
                    log.redundancy_reward = 1.0 - log.redundancy
                    reward = log.redundancy_reward
                elif phyid_reward_type_enum.SYNERGY_REDUNDANCY == self.config.reward_type:
                    log.synergy_redundancy_reward = 0.5 + (log.synergy - log.redundancy) / 2.0
                    reward = log.synergy_redundancy_reward
            
            except Exception as e:
                logging.exception("An exception occurred [calculate_phyid_reward]")                        
                traceback.print_exc()                        
                print(f"[Exception] calculate_phyid_reward: {e}")
                
            rewards.append(reward)
        
        self.get_logger().write_to_log_file()
        return rewards

    @torch.inference_mode()
    def generate_binary_timeseries(self, log: PID_training_log_entity) -> PID_training_log_entity:
        if log.prompt is None or log.completion is None or len(log.completion) == 0:
            return log
        
        trainer = self.get_trainer()
        model = trainer.model
        tokenizer = trainer.processing_class
        
        prompt_embedding = None
        completion_embedding = None
        attention_score = None
        attention_weight = None
        embedding_for_pca = None
        try:
            refine_prompt = self.clean_prompt_for_phi(log.prompt)
            prompt_embedding = super().extract_representation(refine_prompt, model, tokenizer, self.config.layer_type)
            completion_embedding = super().extract_representation(log.completion, model, tokenizer, self.config.layer_type)
            attention_score = self.attention_score(prompt_embedding, completion_embedding)
            attention_weight = self.attention_weight(attention_score)
            embedding_for_pca = np.concatenate(tuple(np.matmul(attention_weight, prompt_embedding)), axis=1,)
            pca = decomposition.PCA(n_components = self.config.reduced_dimention, random_state = 42)
            pca.fit(embedding_for_pca)
            time_series = pca.transform(embedding_for_pca)
            
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
            del prompt_embedding, completion_embedding, attention_score, attention_weight, embedding_for_pca
            gc.collect()

        return log
        



    