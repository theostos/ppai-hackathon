from .training.trainer import load_trainer


EXP_PATH = "results"
trainer, model, model_dir = load_trainer(EXP_PATH, 'logs')
trainer.train()