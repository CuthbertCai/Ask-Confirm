import sys
sys.path.append('/data/home/cuthbertcai/programs/DiaVL/')
import torch
import random
from torch.utils.data import DataLoader
import numpy as np

from train.trainer_img_attr import ImageAttributeTrainer
from utils.vocab import Vocabulary
from utils.config import get_test_config
from datasets.vg import vg
from datasets.loader import region_loader, region_collate_fn

def main(config):
    train_db = vg(config, 'test')
    trainer = ImageAttributeTrainer(config)
    all_img_logits = []
    trainer.net.eval()
    loaddb = region_loader(train_db)
    loader = DataLoader(loaddb, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, collate_fn=region_collate_fn)
    with torch.no_grad():
        for cnt, batched in enumerate(loader):
            region_feats, _, _, obj_inds = trainer.batch_data(batched)
            logits = trainer.net(region_feats)
            all_img_logits.append(logits)

        all_img_logits = torch.cat(all_img_logits, 0)

    all_img_logits = all_img_logits.cpu().numpy()

    np.save('./data/caches/vg_test_img_logits.npy', all_img_logits)

if __name__ == '__main__':
    config, unparsed = get_test_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    main(config)
