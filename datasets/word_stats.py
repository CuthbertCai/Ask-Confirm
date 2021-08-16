import sys
sys.path.append('../')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import os.path as osp
import pickle

from datasets.vg import vg
from datasets.loader import region_loader
from utils.config import get_train_config
from utils.vocab import Vocabulary

stop_words = stopwords.words('english')
for w in ['!', ',' ,'.' ,'?' ,'-s' ,'-ly' ,'</s> ', 's', "'s"]:
    stop_words.append(w)

def stat(loaddbs, config):
    # word2idx = {}
    # count = 0
    # words = set()
    # # build word2idx
    # for loaddb in loaddbs:
    #     for scene in loaddb:
    #         captions = scene[1]
    #         for caption in captions:
    #             word = [w for w in word_tokenize(caption)]
    #             word_new = set()
    #             for w in word:
    #                 if w not in stop_words:
    #                     word_new.add(w)
    #             words = words | word_new
    # print(words)
    # word_num = len(words)
    # print(word_num)
    #
    # for w in words:
    #     word2idx[w] = count
    #     count += 1
    with open(osp.join('/data/home/cuthbertcai/programs/DiaVL/data/caches/vg_vocab_14284.pkl'), 'rb') as fid:
        lang_vocab = pickle.load(fid)

    word_num = len(lang_vocab)
    print(lang_vocab.word2idx)
    print(word_num)

    idx2obj = np.zeros((word_num, config.n_categories))

    for loaddb in loaddbs:
        for scene in loaddb:
            captions, obj, obj_ind = scene[1], scene[5], scene[6]
            words = []
            for caption in captions:
                word = [w for w in word_tokenize(caption)]
                words.extend(word)
            for word in words:
                if word in lang_vocab.word2idx.keys():
                    idx2obj[lang_vocab.word2idx[word]][obj_ind] += 1
    file_path = '/data/home/cuthbertcai/programs/DiaVL/data/caches/vg_word_stat.npy'
    np.save(file_path, idx2obj)
def main():
    config, unparsed = get_train_config()
    loaddbs = []
    for split in ['train']:
        db = vg(config, split)
        loaddb = region_loader(db)
        loaddbs.append(loaddb)
    stat(loaddbs, config)

if __name__ == '__main__':
    main()
