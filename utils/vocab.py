#!/usr/bin/env python

import nltk, pickle
from collections import Counter
from visual_genome.local import get_all_image_data, get_all_region_descriptions
import json, os, argparse
import os.path as osp


this_dir = osp.dirname(__file__)


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


# def from_coco_json(path):
#     coco = COCO(path)
#     ids = coco.anns.keys()
#     captions = []
#     for i, idx in enumerate(ids):
#         captions.append(str(coco.anns[idx]['caption']))

#     return captions


# def from_vg_json(vg_dir):
#     all_regions = get_all_region_descriptions(vg_dir)
#     captions = []
#     for i in range(len(all_regions)):
#         image_regions = all_regions[i]
#         image_captions = [x.phrase.lower().encode('utf-8').decode('utf-8') for x in image_regions]
#         captions = captions + image_captions
#     return captions


# def build_coco_vocab(data_path, data_name, threshold):
#     counter = Counter()
#     # use both the training and validation splits
#     jsons = ['annotations/captions_train2014.json', 'annotations/captions_val2014.json']
#     for path in jsons:
#         full_path = osp.join(osp.join(data_path, data_name), path)
#         captions = from_coco_json(full_path)
#         for i, caption in enumerate(captions):
#             tokens = nltk.tokenize.word_tokenize(caption.lower().encode('utf-8').decode('utf-8'))
#             counter.update(tokens)
#             if i % 1000 == 0:
#                 print("[%d/%d] tokenized the captions." % (i, len(captions)))

#     # Discard if the occurrence of the word is less than min_word_cnt.
#     words = [word for word, cnt in counter.items() if cnt >= threshold]

#     # Create a vocab wrapper and add some special tokens.
#     vocab = Vocabulary()
#     vocab.add_word('<pad>')
#     vocab.add_word('<start>')
#     vocab.add_word('<end>')
#     vocab.add_word('<unk>')

#     # Add words to the vocabulary.
#     for i, word in enumerate(words):
#         vocab.add_word(word)
#     return vocab


# def build_vg_vocab(data_path, data_name, threshold):
#     counter = Counter()
#     vg_dir = osp.join(data_path, data_name)
#     captions = from_vg_json(vg_dir)
#     for i, caption in enumerate(captions):
#         tokens = nltk.tokenize.word_tokenize(caption.lower().encode('utf-8').decode('utf-8'))
#         counter.update(tokens)
#         if i % 1000 == 0:
#             print("[%d/%d] tokenized the captions." % (i, len(captions)))
        
#     # Discard if the occurrence of the word is less than min_word_cnt.
#     words = [word for word, cnt in counter.items() if cnt >= threshold]

#     # Create a vocab wrapper and add some special tokens.
#     vocab = Vocabulary()
#     vocab.add_word('<pad>')
#     vocab.add_word('<start>')
#     vocab.add_word('<end>')
#     vocab.add_word('<unk>')

#     # Add words to the vocabulary.
#     for i, word in enumerate(words):
#         vocab.add_word(word)
#     return vocab


# def main(data_path, data_name):
#     # vocab = build_coco_vocab(data_path, data_name, threshold=4)
#     vocab = build_vg_vocab(data_path, data_name, threshold=10)
#     print('vocab: ', len(vocab))
#     with open('%s_vocab.pkl' % data_name, 'wb') as f:
#         pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
#     print("Saved vocabulary file to ", './%s_vocab.pkl' % data_name)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', default=osp.join(this_dir, '..', 'data'))
#     parser.add_argument('--data_name', default='coco')
#     opt = parser.parse_args()
#     main(opt.data_path, opt.data_name)
