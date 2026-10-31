from src.training.WMS_grpo_trainer import WMS_grpo_trainer
from src.logger.training.WMS_training_logger import WMS_training_logger
from trl import GRPOConfig, ModelConfig
from src.datasets.math.gsm8k.gsm8k_dataset import gsm8k_dataset
from src.datasets.dataset_config import dataset_config
from src.utils.enums_class import training_type_enum, phyid_layer_type_enum, phyid_threashold_type_enum
from src.training.training_config import training_config


class WMS_grpo_trainer_settings_1(WMS_grpo_trainer): 

    def __init__(self):
        config = training_config(
            model_name="Qwen/Qwen3-0.6B",
            training_type = training_type_enum.INFORMATION_DECOMPOSITION_ACCURACY_REWARD, 
            layer_type = phyid_layer_type_enum.LAST,
            threashold_type = phyid_threashold_type_enum.AVERAGE,
        )
        
        super().__init__(config)

    def get_dataset(self) -> gsm8k_dataset:
        if self.dataset is None:
            config = dataset_config(self.config.model_name)
            config.set_max_test_dataset_size(100)
            self.dataset = gsm8k_dataset(config)
        return self.dataset

    def get_model_config(self):
        if self.model_config is None:
            self.model_config = ModelConfig(
                model_name_or_path = self.config.model_name,
                attn_implementation="flash_attention_2",
                use_peft=True,
                lora_r=512,
                lora_alpha=512,
                load_in_4bit=True,
            )
        return self.model_config

    def get_training_args(self):
        
        if self.training_args is None:
            self.training_args = GRPOConfig(
                output_dir="live_logs/settings_1",
                learning_rate=3e-6,
                lr_scheduler_type="cosine",
                logging_steps=10,
                max_steps=1200,
                per_device_train_batch_size=4,      
                gradient_accumulation_steps=4,
                num_generations=2,                      
                per_device_eval_batch_size=16,
                num_generations_eval=1,                      
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                bf16=True,
                max_completion_length=1024, 
                beta=0.001,
                warmup_ratio=0.0,

                report_to=['tensorboard'],
                logging_dir='live_logs/settings_1/tb_logs',  
                eval_strategy="steps",  
                eval_steps=50,
                save_steps=50
            )

        return self.training_args
    
    def get_logger(self):
        if self.logger is None:
            self.logger = WMS_training_logger(log_file_name = 'live_logs/settings_1/settings_1.csv')

        return self.logger


t = WMS_grpo_trainer_settings_1()
t.train()
