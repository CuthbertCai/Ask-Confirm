import sys
sys.path.append('./')
import random
import numpy as np

import torch

from train.trainer_txt_img_matching_global import TextImageMatchingGlobalTrainer
from utils.utils import *
from utils.config import get_test_config
from utils.vocab import Vocabulary
from datasets.vg import vg

def test_model(config):
    test_db = vg(config, 'test')
    trainer = TextImageMatchingGlobalTrainer(config)
    trainer.test(test_db)

if __name__ == '__main__':
    config, unparsed = get_test_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    test_model(config)