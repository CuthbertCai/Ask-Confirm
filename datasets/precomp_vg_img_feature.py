import sys
sys.path.append('../')
import torch
import random
from torch.utils.data import DataLoader
import numpy as np

from train.trainer_txt_img_matching import TextImageMatchingTrainer
from utils.vocab import Vocabulary
from utils.config import get_test_config
from datasets.vg import vg
from datasets.loader import region_loader, region_collate_fn

def main(config):
    train_db = vg(config, 'test')
    trainer = TextImageMatchingTrainer(config)
    all_img_feats = []
    trainer.net.eval()
    loaddb = region_loader(train_db)
    loader = DataLoader(loaddb, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, collate_fn=region_collate_fn)
    with torch.no_grad():
        for cnt, batched in enumerate(loader):
            sent_inds, sent_msks, region_feats = trainer.batch_data(batched)
            img_feats, _ = trainer.net(sent_inds, sent_msks, region_feats)
            all_img_feats.append(img_feats)

        all_img_feats = torch.cat(all_img_feats, 0)

    all_img_feats = all_img_feats.cpu().numpy()

    np.save('./data/caches/vg_test_img_feat.npy', all_img_feats)

if __name__ == '__main__':
    config, unparsed = get_test_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    main(config)
