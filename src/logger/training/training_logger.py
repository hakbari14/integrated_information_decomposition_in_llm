from src.logger.logger import logger

class training_logger(logger): 

    def __init__(self, log_file_name) -> None:
        super().__init__(log_file_name)

    def get_log_list(self, trainer_global_step): 
        return list(filter(lambda x: x.trainer_global_step == trainer_global_step , self.buffer))

