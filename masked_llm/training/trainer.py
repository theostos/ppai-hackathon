import os

from transformers import TrainingArguments
from trl import SFTTrainer
import yaml

from .data_collator import DataCollatorForCompletionOnlyLMask
from .dataset import load_dataset
from .model import load_tokenizer, load_model
from .prompt import formatting_prompts

def load_trainer(export_path, log_path):
    current_script_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_script_dir, 'config_trainer.yaml')
    with open(file_path, 'r', encoding='utf-8') as file:
        config_trainer = yaml.safe_load(file)
    
    model = load_model()
    tokenizer = load_tokenizer()
    dataset = load_dataset()
    collator = DataCollatorForCompletionOnlyLMask("<|start_header_id|>assistant<|end_header_id|>", tokenizer=tokenizer)

    training_arguments = TrainingArguments(
        **config_trainer,
        output_dir= export_path,
        logging_dir=log_path,
        report_to='tensorboard'
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=collator,
        formatting_func=formatting_prompts
    )
    return trainer, model, current_script_dir