import sys
sys.path.append('/data/home/cuthbertcai/programs/DiaVL/')
import torch
import random
from torch.utils.data import DataLoader
import numpy as np
from nltk.tokenize import word_tokenize

from train.trainer_txt_img_matching import TextImageMatchingTrainer
from utils.vocab import Vocabulary
from utils.config import get_test_config
from utils.utils import *
from datasets.vg import vg

def main(config):
    train_db = vg(config, 'test')
    trainer = TextImageMatchingTrainer(config)
    trainer.net.eval()
    attributes = sorted(train_db.class_to_ind.keys(), key=lambda k: train_db.class_to_ind[k])
    attributes_idx = [train_db.class_to_ind[k] for k in attributes]
    # for attr in attributes:
    #     print(attr, train_db.class_to_ind[attr])
    word_inds = []
    lengths = []
    for t in attributes:
        tokens = [w for w in word_tokenize(t)]
        word_ind = [train_db.lang_vocab(w) for w in tokens]
        word_ind.append(config.EOS_idx)
        lengths.append(len(word_ind))
        word_inds.append(torch.Tensor(word_ind))
    max_length = max(lengths)
    new_word_inds = torch.zeros(len(word_inds), max_length).long()
    new_word_msks = torch.zeros(len(word_inds), max_length).long()
    for i in range(len(lengths)):
        new_word_inds[i, :lengths[i]] = word_inds[i]
        new_word_msks[i, :lengths[i]] = 1.0
    if config.cuda:
        new_word_inds = new_word_inds.cuda(non_blocking=True)
        new_word_msks = new_word_msks.cuda(non_blocking=True)
    _, attr_feat, _ = trainer.net.txt_enc(new_word_inds, new_word_msks)
    attr_feat = attr_feat.detach()
    if config.l2_norm:
        attr_feat = l2norm(attr_feat)
    if config.cuda:
        attr_feat = attr_feat.cpu().numpy()
    else:
        attr_feat = attr_feat.numpy()
    id2attr = {}
    for i,idx in enumerate(attributes_idx):
        if idx not in id2attr:
            id2attr[idx] = [attr_feat[i]]
        else:
            id2attr[idx].append(attr_feat[i])
    attr_feat = np.zeros((config.n_categories, config.n_feature_dim))
    for key in id2attr.keys():
        attrs = np.array(id2attr[key])
        attr = np.mean(attrs, axis=0)
        attr_feat[key] = attr
    np.save('./data/caches/vg_attr_feat.npy', attr_feat)

if __name__ == '__main__':
    config, unparsed = get_test_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    main(config)



