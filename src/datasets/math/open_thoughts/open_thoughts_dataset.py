from src.datasets.math.math_dataset_handler import math_dataset_handler
from src.datasets.dataset_config import dataset_config
from src.utils.enums_class import llm_pipeline_type_enum
from datasets import Dataset
from datasets import load_dataset
from src.datasets.math.utils.evaluate_utils import extract_last_boxed, use_math_verify
import re

class open_thoughts_dataset(math_dataset_handler): 

    def __init__(self, config):
        super().__init__(config)
        self.dataset_id = "anonym-submit-paper/Orig-R1-Thoughts-correct"
        self.dataset = load_dataset(self.dataset_id)
        correct_dataset = self.dataset.filter(lambda x: self.filter_dataset(x))
        
        dataset = correct_dataset["train"].train_test_split(test_size=0.1)
        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]
        self.instruction = self.prompt_config.get('Math', 'open_thought')        
        self.force_generate_answer_text = '\\boxed{'
    
    def generate_model_prompt(self, x):
        question = x['problem']
        solution = x['solution']
        problem_id = x['ProblemIdx']

        final_answer = self.final_answer_extraction('', solution, '')
        r1_prefix = [{"role": "system", "content": self.instruction},
                     {"role": "user", "content": question}
                    ]

        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": final_answer,
                "problem_id": problem_id,
                "question": question,
                }

    def final_answer_extraction(self, prompt, completion, target):
        return extract_last_boxed(completion)

    def extract_final_answer(self, completion):
        patterns = [
            r"Final\s*Answer\s*:\s*\n(.*?)(?:\n\s*\n|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, completion, re.IGNORECASE)
            if not match: continue
            return match.group(1)

        answer_to_confidence = self.extract_between_final_answer_to_confidence(completion)
        final_answer = extract_last_boxed(answer_to_confidence)
        if final_answer is not None:
            return final_answer
        
        return answer_to_confidence
    
    def filter_dataset(self, x):
        if x['correct'] != True:
            return False
        if self.config.get_max_completion_length() is not None and x['generated_token_count'] > self.config.get_max_completion_length():
            return False
        return True
    
    
