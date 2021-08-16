import sys
sys.path.append('/home/cgy/programs/DiaVL')
import numpy as np
import torch

from train.trainer_ppo_coherence import PPOTrainer
from utils.config import get_test_config
from utils.vocab import Vocabulary

def test_model(config):
    trainer = PPOTrainer(config, 'test')
    trainer.test()
    # trainer.test_once()

if __name__ == '__main__':
    config, unparsed = get_test_config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    test_model(config)
