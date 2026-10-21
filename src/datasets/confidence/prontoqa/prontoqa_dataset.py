from src.datasets.dataset_handler import dataset_handler
from src.datasets.dataset_config import dataset_config
from datasets import Dataset
from datasets import load_dataset
import re

class prontoqa_dataset(dataset_handler): 

    def __init__(self, config: dataset_config):
        super().__init__(config)
        
        self.dataset_id = 'renma/ProntoQA'
        self.dataset: Dataset = load_dataset(self.dataset_id)
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.dataset['validation']

        if config.get_ratio_test_dataset_size() is not None: 
            self.test_dataset = self.test_dataset.train_test_split(test_size=config.get_ratio_test_dataset_size(), seed=42, shuffle=True)['test']

    
    def final_answer_extraction(self, prompt, solution, target):
        # The last one the model wrote: an earlier one may only repeat the instruction.
        # A capital letter, so "final answer: a contradiction" is not read as A.
        matches = re.findall(r'(?i:final\s+answer)[\s:*]*\(?([AB])\b', solution)
        return matches[-1] if matches else None

        
    def generate_model_prompt(self, x):
        unique_id: str = x['id']
        question: str = x['question']
        context: str = x['context']
        choices_text: str = x['options']
        target = x['answer']
        
        prompt = (
            "You are given a logical reasoning problem.\n\n"
            "Use only the facts and rules provided in the context below. "
            "Determine whether the statement in the question is True or False.\n\n"
            "Reason step by step by applying the rules logically. "
            "Do not use any external knowledge or assumptions.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{choices_text}\n\n"
            "Think step by step and derive the answer from the given facts and rules.\n"
            "At the end of your response, output the final answer exactly as:\n"
            "Final Answer: A\n"
            "or\n"
            "Final Answer: B"
        )

        r1_prefix = [
            {"role": "user",
                "content": prompt
                },
        ]
        
        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True), 
                "target": target,
                "question": question,
                "problem_id": unique_id
                }


    def generate_wrong_answer(self, latex_expr:str) -> str:
        return None

