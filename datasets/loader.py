#!/usr/bin/env python

import os, sys, cv2, json

sys.path.append('../')
import random, pickle, math
import numpy as np
import os.path as osp
from PIL import Image
from time import time
from copy import deepcopy
from glob import glob
from nltk.tokenize import word_tokenize

from utils.config import get_train_config, get_test_config
from utils.utils import *
from datasets.caption_template import *
from datasets.vg import vg
from utils.vocab import Vocabulary

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class caption_loader(Dataset):
    def __init__(self, imdb):
        self.cfg = imdb.cfg
        self.db = imdb
        # normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # t_list = []
        # # if self.db.split in ['val', 'test']:
        # #     t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        # # else:
        # #     t_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        # t_list = [transforms.Resize((224, 224))]
        # t_end = [transforms.ToTensor(), normalizer]
        # self.transform = transforms.Compose(t_list + t_end)

    def __len__(self):
        return len(self.db.scenedb)

    def __getitem__(self, scene_index):
        scene = self.db.scenedb[scene_index]
        # image
        image_index = scene['image_index']
        resnet_path = self.db.resnet_path_from_index(image_index)
        image = torch.from_numpy(pickle_load(resnet_path)).squeeze().float()
        # image_path = self.db.color_path_from_index(image_index)
        # image = Image.open(image_path).convert('RGB')
        # image = self.transform(image)

        # captions
        if self.db.name == 'coco':
            all_captions = scene['captions']
        else:
            all_meta_regions = [scene['regions'][x] for x in sorted(list(scene['regions'].keys()))]
            all_captions = [x['caption'] for x in all_meta_regions]
        if self.db.split in ['val', 'test']:
            captions = tuple(all_captions[:self.cfg.max_turns])
            if self.cfg.negation > 0:
                temps = list(captions)[:(self.cfg.max_turns - 2 * self.cfg.negation)]
                negative_objects = scene['negative_objects']
                for x in sorted(list(negative_objects.keys()))[:self.cfg.negation]:
                    name = negative_objects[x]['name']
                    positive_cap = positive_object_templates[
                                       np.random.permutation(range(len(positive_object_templates)))[0]] % name
                    negative_cap = negative_object_templates[
                                       np.random.permutation(range(len(negative_object_templates)))[0]] % name
                    temps.append([positive_cap, negative_cap])
                captions = []
                for x in range(self.cfg.max_turns):
                    new_indices = np.random.permutation(range(len(temps)))
                    temps = [temps[y] for y in new_indices]
                    if isinstance(temps[0], list):
                        captions.append(temps[0][0])
                        temps[0] = temps[0][1]
                    else:
                        captions.append(temps[0])
                        temps = temps[1:]
                assert (len(temps) == 0)
                captions = tuple(captions)
        else:
            num_captions = len(all_captions)
            caption_inds = np.random.permutation(range(num_captions))
            captions = tuple([all_captions[x] for x in caption_inds[:self.cfg.max_turns]])
            if self.cfg.negation > 0:
                temps = list(captions)[:(self.cfg.max_turns - 2 * self.cfg.negation)]
                negative_objects = scene['negative_objects']
                for x in sorted(list(negative_objects.keys()))[:self.cfg.negation]:
                    name = negative_objects[x]['name']
                    positive_cap = positive_object_templates[
                                       np.random.permutation(range(len(positive_object_templates)))[0]] % name
                    negative_cap = negative_object_templates[
                                       np.random.permutation(range(len(negative_object_templates)))[0]] % name
                    temps.append([positive_cap, negative_cap])
                captions = []
                for x in range(self.cfg.max_turns):
                    new_indices = np.random.permutation(range(len(temps)))
                    temps = [temps[y] for y in new_indices]
                    if isinstance(temps[0], list):
                        captions.append(temps[0][0])
                        temps[0] = temps[0][1]
                    else:
                        captions.append(temps[0])
                        temps = temps[1:]
                assert (len(temps) == 0)
                captions = tuple(captions)

        sent_inds = []
        for i in range(self.cfg.max_turns):
            tokens = [w for w in word_tokenize(captions[i])]
            # tokens = further_token_process(tokens)
            word_inds = [self.db.lang_vocab(w) for w in tokens]
            word_inds.append(self.cfg.EOS_idx)
            sent_inds.append(torch.Tensor(word_inds))
        sent_inds = tuple(sent_inds)
        return image, sent_inds, captions, image_index, scene_index


def caption_collate_fn(data):
    images, sent_inds, captions, image_indices, scene_indices = zip(*data)
    images = torch.stack(images, 0)

    lengths = [len(sent_inds[i][j]) for i in range(len(sent_inds)) for j in range(len(sent_inds[0]))]
    max_length = max(lengths)
    new_sent_inds = torch.zeros(len(sent_inds), len(sent_inds[0]), max_length).long()
    new_sent_msks = torch.zeros(len(sent_inds), len(sent_inds[0]), max_length).long()
    for i in range(len(sent_inds)):
        for j in range(len(sent_inds[0])):
            end = len(sent_inds[i][j])
            new_sent_inds[i, j, :end] = sent_inds[i][j]
            new_sent_msks[i, j, :end] = 1

    entry = {
        'images': images,
        'sent_inds': new_sent_inds,
        'sent_msks': new_sent_msks,
        'captions': captions,
        'image_inds': image_indices,
        'scene_inds': scene_indices
    }

    return entry


class region_loader(Dataset):
    def __init__(self, imdb):
        self.cfg = imdb.cfg
        self.db = imdb
        self.obj_to_ind = imdb.class_to_ind
        # print(self.obj_to_ind)

    def __len__(self):
        return len(self.db.scenedb)

    def __getitem__(self, scene_index):
        scene = self.db.scenedb[scene_index]
        # print(scene['objects'])
        image_index = scene['image_index']

        # objects and attributes
        objs = [scene['objects'][obj]['name'] for obj in scene['objects'] if scene['objects'][obj]['name'] in self.obj_to_ind.keys()]
        atts = [scene['objects'][obj]['atts'] for obj in scene['objects'] if scene['objects'][obj]['name'] in self.obj_to_ind.keys()]
        objs = list(set(objs))
        objs_inds = list(set([self.obj_to_ind[obj] for obj in objs]))

        # region features
        region_path = self.db.region_path_from_index(image_index)
        with open(region_path, 'rb') as fid:
            regions = pickle.load(fid, encoding='latin1')
        region_boxes = torch.from_numpy(regions['region_boxes']).float()
        region_feats = torch.from_numpy(regions['region_feats']).float()
        region_clses = torch.from_numpy(regions['region_clses']).long()

        # captions
        if self.db.name == 'coco':
            all_captions = scene['captions']
        else:
            all_meta_regions = [scene['regions'][x] for x in sorted(list(scene['regions'].keys()))]
            all_captions = [x['caption'] for x in all_meta_regions]

        width = scene['width']
        height = scene['height']
        if self.db.split in ['val', 'test']:
            captions = all_captions[:self.cfg.max_turns]
            if self.cfg.negation > 0:
                temps = list(captions)[:(self.cfg.max_turns - 2 * self.cfg.negation)]
                negative_objects = scene['negative_objects']
                for x in sorted(list(negative_objects.keys()))[:self.cfg.negation]:
                    name = negative_objects[x]['name']
                    positive_cap = positive_object_templates[
                                       np.random.permutation(range(len(positive_object_templates)))[0]] % name
                    negative_cap = negative_object_templates[
                                       np.random.permutation(range(len(negative_object_templates)))[0]] % name
                    temps.append([positive_cap, negative_cap])
                captions = []
                for x in range(self.cfg.max_turns):
                    new_indices = np.random.permutation(range(len(temps)))
                    temps = [temps[y] for y in new_indices]
                    if isinstance(temps[0], list):
                        captions.append(temps[0][0])
                        temps[0] = temps[0][1]
                    else:
                        captions.append(temps[0])
                        temps = temps[1:]
                assert (len(temps) == 0)
                # captions = tuple(captions)
        else:
            num_captions = len(all_captions)
            caption_inds = np.random.permutation(range(num_captions))
            captions = [all_captions[x] for x in caption_inds[:self.cfg.max_turns]]
            if self.cfg.negation > 0:
                temps = list(captions)[:(self.cfg.max_turns - 2 * self.cfg.negation)]
                negative_objects = scene['negative_objects']
                for x in sorted(list(negative_objects.keys()))[:self.cfg.negation]:
                    name = negative_objects[x]['name']
                    positive_cap = positive_object_templates[
                                       np.random.permutation(range(len(positive_object_templates)))[0]] % name
                    negative_cap = negative_object_templates[
                                       np.random.permutation(range(len(negative_object_templates)))[0]] % name
                    temps.append([positive_cap, negative_cap])
                captions = []
                for x in range(self.cfg.max_turns):
                    new_indices = np.random.permutation(range(len(temps)))
                    temps = [temps[y] for y in new_indices]
                    if isinstance(temps[0], list):
                        captions.append(temps[0][0])
                        temps[0] = temps[0][1]
                    else:
                        captions.append(temps[0])
                        temps = temps[1:]
                assert (len(temps) == 0)
                # captions = tuple(captions)

        if self.cfg.paragraph_model:
            for i in range(1, len(captions)):
                captions[i] = captions[i - 1] + ';' + captions[i]

        if self.cfg.use_attr:
            captions = aug_txt_with_attr(captions, objs)

        captions = tuple(captions)

        sent_inds = []
        for i in range(self.cfg.max_turns):
            tokens = [w for w in word_tokenize(captions[i])]
            # tokens = further_token_process(tokens)
            word_inds = [self.db.lang_vocab(w) for w in tokens]
            word_inds.append(self.cfg.EOS_idx)
            sent_inds.append(torch.Tensor(word_inds))
        sent_inds = tuple(sent_inds)

        obj2word_inds = []
        for obj in objs:
            token = [w for w in word_tokenize(obj)]
            word_ind = [self.db.lang_vocab(w) for w in token]
            word_ind.append(self.cfg.EOS_idx)
            obj2word_inds.append(torch.Tensor(word_ind))
        obj2word_inds = tuple(obj2word_inds)
        return sent_inds, captions, region_boxes, region_feats, region_clses, objs, objs_inds, obj2word_inds, atts, width, height, image_index, scene_index


def region_collate_fn(data):
    sent_inds, captions, region_boxes, region_feats, region_clses, objs, objs_inds, obj2word_inds, atts, width, height, image_indices, scene_indices = zip(
        *data)

    # regions
    lengths = [region_boxes[i].size(0) for i in range(len(region_boxes))]
    max_length = max(lengths)

    new_region_boxes = torch.zeros(len(region_boxes), max_length, region_boxes[0].size(-1)).float()
    new_region_feats = torch.zeros(len(region_feats), max_length, region_feats[0].size(-1)).float()
    new_region_clses = torch.zeros(len(region_clses), max_length).long()
    new_region_masks = torch.zeros(len(region_clses), max_length).long()

    for i in range(len(region_boxes)):
        end = region_boxes[i].size(0)
        new_region_boxes[i, :end] = region_boxes[i]
        new_region_feats[i, :end] = region_feats[i]
        new_region_clses[i, :end] = region_clses[i]
        new_region_masks[i, :end] = 1.0

    # captions
    lengths = [len(sent_inds[i][j]) for i in range(len(sent_inds)) for j in range(len(sent_inds[0]))]
    max_length = max(lengths)
    new_sent_inds = torch.zeros(len(sent_inds), len(sent_inds[0]), max_length).long()
    new_sent_msks = torch.zeros(len(sent_inds), len(sent_inds[0]), max_length).long()
    for i in range(len(sent_inds)):
        for j in range(len(sent_inds[0])):
            end = len(sent_inds[i][j])
            new_sent_inds[i, j, :end] = sent_inds[i][j]
            new_sent_msks[i, j, :end] = 1

    # objects
    lengths = [len(objs_inds[i]) for i in range(len(objs_inds)) ]
    max_length = max(lengths)
    new_objs_inds = torch.zeros(len(objs_inds), max_length).long()
    new_objs_msks = torch.zeros(len(objs_inds), max_length).long()
    for i in range(len(objs_inds)):
        for j in range(len(objs_inds[i])):
            new_objs_inds[i, j] = objs_inds[i][j]
            new_objs_msks[i, j] = 1.0

    # objects to words
    # lengths = [len(obj) for obj in objs]
    # max_length = max(lengths)
    # print(max_length)
    new_obj2word_inds = torch.zeros(len(obj2word_inds), 50, 5).long()
    new_obj2word_msks = torch.zeros(len(obj2word_inds), 50, 5).long()
    for i in range(len(obj2word_inds)):
        for j in range(len(obj2word_inds[i])):
            end = len(obj2word_inds[i][j])
            new_obj2word_inds[i,j,:end] = obj2word_inds[i][j]
            new_obj2word_msks[i,j,:end] = 1.0

    entry = {
        'region_boxes': new_region_boxes,
        'region_feats': new_region_feats,
        'region_masks': new_region_masks,
        'region_clses': new_region_clses,
        'sent_inds': new_sent_inds,
        'sent_msks': new_sent_msks,
        'objs_inds': new_objs_inds,
        'objs_msks': new_objs_msks,
        'obj2word_inds': new_obj2word_inds,
        'obj2word_msks': new_obj2word_msks,
        'captions': captions,
        'objects': objs,
        'attributes': atts,
        'widths': width,
        'heights': height,
        'image_inds': image_indices,
        'scene_inds': torch.Tensor(scene_indices)
    }

    return entry


def test():
    config, unparsed = get_train_config()

    traindb = vg(config, 'train')
    train_loaddb = region_loader(traindb)
    train_loader = torch.utils.data.DataLoader(train_loaddb, batch_size=10, shuffle=False, collate_fn=region_collate_fn)
    print(len(train_loaddb))
    for data in train_loader:
        for key in data:
            print(key, data[key].shape if isinstance(data[key], torch.Tensor) else data[key])
            # pass
        # print(data['scene_inds'])

        # print(data['region_clses'])

if __name__ == '__main__':
    test()
