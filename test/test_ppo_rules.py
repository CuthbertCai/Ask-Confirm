import sys
sys.path.append('./')
import numpy as np
import torch

from train.trainer_ppo_rules import PPOTrainer
from utils.config import get_test_config
from utils.vocab import Vocabulary

def test_model(config):
    trainer = PPOTrainer(config, 'test')
    trainer.test()

if __name__ == '__main__':
    config, unparsed = get_test_config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    test_model(config)