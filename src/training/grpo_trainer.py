from src.calculator.PIDCalculator import PIDCalculator, PIDResult
from src.logger.training.training_log_entity import training_log_entity
from src.utils.enums_class import training_type_enum, phyid_layer_type_enum, phyid_threashold_type_enum, phyid_reward_type_enum
from src.training.training_config import training_config
from trl import GRPOTrainer, get_peft_config
from abc import ABC, abstractmethod
from sklearn import decomposition
import torch
import numpy as np
import re 

class grpo_trainer(ABC): 

    def __init__(self, config: training_config) -> None:
        self.config = config
        self.dataset = None
        self.model_config = None
        self.training_args = None
        self.trainer = None
        self.logger = None
        self.calculator = PIDCalculator()

    def train(self):
        trainer = self.get_trainer()
        trainer.model.config.use_cache = False
        trainer.train()

    def get_trainer(self):
        if self.trainer is None:
            train_dataset, eval_dataset = self.get_dataset().preprocess_dataset()
            model_config = self.get_model_config()
            
            self.trainer = GRPOTrainer(
                model = self.config.model_name,
                reward_funcs = self.get_reward_funcs(),
                args = self.get_training_args(),
                train_dataset = train_dataset,
                eval_dataset = eval_dataset,
                peft_config = get_peft_config(model_config),
            )
            
            self.trainer.model.print_trainable_parameters()
        return self.trainer

    def get_reward_funcs(self):
        if training_type_enum.ACCURACY_REWARD == self.config.training_type: 
            return [self.accuracy_reward]

        if training_type_enum.ACCURACY_REWARD_PHY_ID  == self.config.training_type: 
            return [self.accuracy_reward, self.calculate_phyid_reward]

        return []

    @torch.inference_mode()
    def calculate_phyid_reward(self, completions, target=None, tokenizer=None, **kwargs):
        split_list = kwargs.get("split")     
        sample_ids = kwargs.get("sample_id") 
        problem_ids = kwargs.get("problem_id", None)
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        questions = kwargs.get("question")     
        trainer_state = kwargs.get("trainer_state", None)
        trainer_global_step = trainer_state.global_step
        
        log_list: list[training_log_entity] = self.get_log_list(trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target)
    
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
                result: PIDResult = self.calculator.compute(log.time)
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
                    log.synergy_redundancy_reward = log.synergy - log.redundancy
                    reward = log.synergy_redundancy_reward
            except Exception as e:
                print(f"[Exception] calculate_phyid_reward: {e}")
                
            rewards.append(reward)
        
        self.get_logger().write_to_log_file()
        return rewards

    def accuracy_reward(self, completions, target, **kwargs):
        rewards = []
        split_list = kwargs.get("split")     
        sample_ids = kwargs.get("sample_id") 
        problem_ids = kwargs.get("problem_id", None)
        prompts = kwargs.get("prompts") or kwargs.get("prompt") or kwargs.get("inputs")
        questions = kwargs.get("question")     
        trainer_state = kwargs.get("trainer_state", None)
        trainer_global_step = trainer_state.global_step

        log_list: list[training_log_entity] = self.get_log_list(trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target)
        for log in log_list: 
            rewards.append(log.accuracy_reward)

        if training_type_enum.ACCURACY_REWARD == self.config.training_type:
            self.get_logger().write_to_log_file()

        return rewards

    def generate_binary_timeseries(self, log: training_log_entity) -> training_log_entity:
        if log.prompt is None or log.completion is None or len(log.completion) == 0:
            return log
        
        trainer = self.get_trainer()
        model = trainer.model
        tokenizer = trainer.processing_class

        try:
            refine_prompt = self.clean_prompt_for_phi(log.prompt)
            prompt_embedding = self.extract_representation(refine_prompt, model, tokenizer, self.config.layer_type)
            completion_embedding = self.extract_representation(log.completion, model, tokenizer, self.config.layer_type)
            attention_score = self.attention_score(prompt_embedding, completion_embedding)
            attention_weight = self.attention_weight(attention_score)
            completion_embedding_for_pca = np.concatenate(tuple(np.matmul(attention_weight, prompt_embedding)), axis=1,)
            pca = decomposition.PCA(n_components = self.config.reduced_dimention, random_state = 42)
            pca.fit(completion_embedding_for_pca)
            time_series = pca.transform(completion_embedding_for_pca)
            
            if phyid_threashold_type_enum.AVERAGE == self.config.threashold_type: 
                avgs = np.mean(time_series, axis=0)
            elif phyid_threashold_type_enum.MEDIAN == self.config.threashold_type: 
                avgs = np.median(time_series, axis=0)
            
            time_series_copy = np.copy(time_series)
            for i in range(len(avgs)):
                time_series[np.where(time_series_copy[:, i] >= avgs[i]), i] = 1
                time_series[np.where(time_series_copy[:, i] < avgs[i]), i] = 0

            log.time_series = time_series 
        except Exception as e:
            print(f"[Exception] generate_binary_timeseries: {e}")

        return log
        
    def get_log_list(self, trainer_global_step, split_list, sample_ids, problem_ids, questions, prompts, completions, target) -> list[training_log_entity]:
        log_list = self.get_logger().get_log_list(trainer_global_step)
        
        if len(log_list) > 0:
            return log_list

        log_list: list[training_log_entity] = []
        for i, (completion, gt, question) in enumerate(zip(completions, target, questions)):
            split = split_list[i]
            sample_ID = sample_ids[i]
            problem_id = problem_ids[i] if problem_ids is not None else None
            prompt = prompts[i]

            log = training_log_entity()
            log.ID = f'{sample_ID}_{i}' 
            log.sample_ID = sample_ID
            log.problem_id = problem_id
            log.split = split
            log.trainer_global_step = trainer_global_step
            log.question = question
            log.prompt = prompt
            log.target = gt
            log.completion = completion
            try:
                answer, target_answer_equal, compared_final_answer = self.get_dataset().extract_and_verify_final_answer(prompt, completion, gt)
                if answer is None:
                    acc_reward = 0.0
                    log.accuracy = False
                else:
                    acc_reward = 1.0 if target_answer_equal else 0.0
                    log.accuracy = target_answer_equal

                log.final_answer = answer
                log.compared_final_answer = compared_final_answer
                log.accuracy_reward = self.config.acurray_reward_coefficient * acc_reward
            except Exception:
                log.accuracy = False
                log.accuracy_reward = 0.0
            
            log_list.append(log)
            self.get_logger().add_to_buffer(log)
        
        return log_list
    
    def extract_representation(self, text: str, model, tokenizer, layer_type: phyid_layer_type_enum) -> np.ndarray:
        inputs = outputs = None
        try:
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors='pt').to(model.device)
                outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True, use_cache=False,)
                num_layers = len(outputs.hidden_states)
                
                if layer_type == phyid_layer_type_enum.SOME:
                    sampled_layers = np.linspace(1, num_layers - 1, num=12)
                    sampled_layers = [int(x) for x in np.round(sampled_layers).tolist()]
                    layer_2_3 = int(round((num_layers - 1) * 2 / 3))
                    if layer_2_3 not in sampled_layers:
                        sampled_layers.append(layer_2_3)
                    layer_indices = sorted(set(sampled_layers))
                elif layer_type == phyid_layer_type_enum.ALL:
                    layer_indices = range(num_layers)
                elif layer_type == phyid_layer_type_enum.LAST:
                    layer_indices = [num_layers - 1]
                else:
                    return None

                # Transfer only requested layers; never concatenate them on GPU.
                hidden = np.concatenate([
                    outputs.hidden_states[i].detach().cpu().float().numpy()
                    for i in layer_indices
                ], axis=0)
                
                # Convert one token's logits at a time, avoiding a full FP32 copy.
                if layer_type == phyid_layer_type_enum.LAST:
                    hidden = hidden[0].copy()
                    
                return hidden
        finally:
            # Also release tensors when extraction raises (including CUDA OOM).
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def attention_score(self, prompt, response, l_mask_spans=None, l_mask_context=None):
        n_l, n_1, D = prompt.shape
        _, n_2, _ = response.shape

        attention_scores = np.matmul(response, prompt.transpose(0, 2, 1))  # (n_l, n_2, n_1)
        attention_scores = attention_scores / np.sqrt(D)

        if (l_mask_spans is not None) and (l_mask_context is not None):
            mask_context = np.zeros(n_1, dtype=bool)
            mask_context[l_mask_context] = True  # Set context indices to True

            mask_span = np.zeros(n_1, dtype=bool)
            mask_span[l_mask_spans] = True  # Set span indices to True

            mask_out_of_both = ~mask_span & ~mask_context
            attention_scores[:, :, mask_context] *= 0.6
            attention_scores[:, :, mask_out_of_both] *= 0.2

        return attention_scores

    def attention_weight(self, attention_scores, axis=-1):
        exps = np.exp(attention_scores - np.max(attention_scores, axis=axis, keepdims=True))
        return exps / np.sum(exps, axis=axis, keepdims=True)

    def clean_prompt_for_phi(self, raw_prompt: str) -> str:
        SYSTEM_BLOCK_RE = re.compile(
            r"<\|im_start\|>\s*system\b.*?<\|im_end\|>\s*", re.DOTALL | re.IGNORECASE
        )
        GENERIC_BLOCK_RE = re.compile(
            r"<\|im_start\|>.*?<\|im_end\|>\s*", re.DOTALL | re.IGNORECASE
        )

        p = SYSTEM_BLOCK_RE.sub("", raw_prompt)

        m = list(re.finditer(
            r"User:\s*(.*?)(?=\n<\|im_start\|>|\nAssistant:|$)",
            p,
            flags=re.DOTALL | re.IGNORECASE
        ))
        if m:
            text = m[-1].group(1)
        else:
            
            text = GENERIC_BLOCK_RE.sub("", p)

        # Remove boilerplate lines that sometimes appear before/around "User:"
        text = re.sub(
            r"^A conversation between.*?User:\s*",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Remove any <think>...</think> that may be embedded in the prompt text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

        return text.strip()

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_model_config(self):
        pass

    @abstractmethod
    def get_training_args(self):
        pass

    @abstractmethod
    def get_logger(self):
        pass




    