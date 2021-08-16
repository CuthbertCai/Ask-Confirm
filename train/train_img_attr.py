import sys
sys.path.append('./')
import random
import numpy as np

import torch

from train.trainer_img_attr import ImageAttributeTrainer
from utils.utils import *
from utils.config import get_train_config
from utils.vocab import Vocabulary
from datasets.vg import vg

def train_model(config):
    train_db = vg(config, 'train')
    val_db = vg(config, 'val')
    test_db = vg(config, 'test')
    trainer = ImageAttributeTrainer(config)
    trainer.train(train_db, val_db, test_db)

if __name__ == '__main__':
    config, unparsed = get_train_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    train_model(config)
