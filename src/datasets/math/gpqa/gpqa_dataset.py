from src.datasets.math.math_dataset_handler import math_dataset_handler
from src.datasets.dataset_config import dataset_config
from src.datasets.math.utils.evaluate_utils import normalize_answer
from src.utils.enums_class import llm_pipeline_type_enum
from datasets import Dataset
from datasets import load_dataset
import re
import pandas as pd


class gpqa_dataset(math_dataset_handler): 

    def __init__(self, config):
        super().__init__(config)
        
        self.dataset_id = "data/gpqa_diamond.csv"
        df = pd.read_csv(self.dataset_id)
        if config.get_ratio_test_dataset_size() is not None: 
            df = df.sample(frac=config.get_ratio_test_dataset_size(), random_state=42)
        ds = Dataset.from_pandas(df).select_columns(['problem', 'answer'])
        self.dataset = ds
        self.dataset = self.dataset.add_column('unique_id', range(len(self.dataset)))
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.dataset

    
    def final_answer_extraction(self, prompt, solution, target):
        return self.multiple_choice_answer(solution, 'ABCD')

        
    def generate_model_prompt(self, x):
        question = x["problem"]
        final_answer = x["answer"]
        problem_id = x["unique_id"]

        q = question.split("Choices:")[0]
        choices_part = question.split("Choices:")[1]
        pattern = r'^([A-D])\.\s*(.+)$'
        matches = re.findall(pattern, choices_part, re.MULTILINE)
        
        prompt = f"What is the correct answer to this question: {q}"
        prompt += f"\n\nChoices:\n(A) {matches[0][1]}\n(B) {matches[1][1]}\n(C) {matches[2][1]}\n(D) {matches[3][1]}"
        prompt += f"\n\nReason step by step, then end your response with: \"The correct answer is (insert answer here)\""
        r1_prefix = [
            {"role": "user",
                "content": prompt
                },
        ]
        
        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True), 
                "target": final_answer,
                "question": question,
                "problem_id": problem_id
                }

