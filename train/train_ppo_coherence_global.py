import sys
sys.path.append('./')
import numpy as np
import torch

from train.trainer_ppo_coherence_global import PPOTrainer
from utils.config import get_train_config
from utils.vocab import Vocabulary

def train_model(config):
    trainer = PPOTrainer(config, 'train')
    trainer.train()

if __name__ == '__main__':
    config, unparsed =  get_train_config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    train_model(config)