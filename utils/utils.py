#!/usr/bin/env python

import os, sys, cv2, json, pickle
import math, PIL
import copy, random, re
from copy import deepcopy
import numpy as np
import os.path as osp
from time import time
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import scipy

import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

this_dir = osp.dirname(__file__)


###########################################################
## Directory
###########################################################

def maybe_create(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)


def prepare_directories(config):
    postfix = datetime.now().strftime("%m%d_%H%M%S")
    model_name = '{}_{}'.format(config.exp_name, postfix)
    config.model_name = model_name
    config.model_dir = osp.join(config.log_dir, model_name)
    maybe_create(config.model_dir)

def prepare_test_directories(config):
    pretrained_path = config.pretrained
    config.model_dir = osp.join('/', *pretrained_path.split('/')[:-2])


###########################################################
## Vocabulary
###########################################################
import string

punctuation_table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))


# print('stop_words: ', stop_words)

def further_token_process(tokens):
    tokens = [w.translate(punctuation_table) for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]
    # TO-DO: determine if stop words should be removed
    # tokens = [w for w in tokens if not w in stop_words]
    return tokens


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def clamp_array(array, min_value, max_value):
    return np.minimum(np.maximum(min_value, array), max_value)


def paint_box(ctx, color, box):
    # bounding box representation: xyxy to xywh
    # box is not normalized
    x = box[0];
    y = box[1]
    w = box[2] - box[0] + 1
    h = box[3] - box[1] + 1

    ctx.set_source_rgb(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    ctx.set_line_width(6)
    ctx.rectangle(x, y, w, h)
    ctx.stroke()

    # ctx.set_operator(cairo.OPERATOR_ADD)
    # ctx.fill()


# def paint_txt(ctx, txt, box):
# Color
# ctx.set_source_rgb(0, 0, 0)
# Font
# font_option = cairo.FontOptions()
# font_option.set_antialias(cairo.Antialias.SUBPIXEL)
# ctx.set_font_options(font_option)
# ctx.select_font_face("Purisa", cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_BOLD)
# ctx.set_font_size(20)
# ctx.set_operator(cairo.OPERATOR_ADD)
# Position
# x = box[0]; y = box[1] + 50
# w = box[2] - box[0] + 1
# h = box[3] - box[1] + 1

# ctx.move_to(x, y)
# ctx.show_text(txt)


def create_squared_image(img, pad_value=None):
    # If pad value is not provided
    if pad_value is None:
        pad_value = np.array([103.53, 116.28, 123.675])

    width = img.shape[1]
    height = img.shape[0]

    # largest length
    max_dim = np.maximum(width, height)
    # anchored at the left-bottom position
    offset_x = 0  # int(0.5 * (max_dim - width))
    offset_y = max_dim - height  # int(0.5 * (max_dim - height))

    output_img = pad_value.reshape(1, 1, img.shape[-1]) * \
                 np.ones((max_dim, max_dim, img.shape[-1]))
    output_img[offset_y: offset_y + height, \
    offset_x: offset_x + width, :] = img

    return output_img.astype(np.uint8), offset_x, offset_y


def create_colormap(num_colors):
    # JET colorbar
    dz = np.arange(1, num_colors + 1)
    norm = plt.Normalize()
    colors = plt.cm.jet(norm(dz))
    return colors[:, :3]


###########################################################
## Data
###########################################################

def pickle_load(path):
    with open(path, 'rb') as fid:
        data_ = pickle.load(fid)
    return data_


def pickle_save(path, data):
    with open(path, 'wb') as fid:
        pickle.dump(data, fid, pickle.HIGHEST_PROTOCOL)


def json_load(path):
    with open(path, 'r') as fid:
        data_ = json.load(fid)
    return data_


def json_save(path, data):
    with open(path, 'w') as fid:
        json.dump(data, fid, indent=4, sort_keys=True)


def pad_sequence(inputs, max_length, pad_val, sos_val=None, eos_val=None, eos_msk=None):
    # cut the input sequence off if necessary
    seq = inputs[:max_length]
    # mask for valid input items
    msk = [1.0] * len(seq)
    # if the length of the inputs is shorter than max_length, pad it with special items provided
    num_padding = max_length - len(seq)
    # pad SOS
    if sos_val is not None:
        if isinstance(sos_val, np.ndarray):
            seq = [sos_val.copy()] + seq
        else:
            seq = [sos_val] + seq
        msk = [1.0] + msk
    # pad EOS
    if eos_val is not None:
        if isinstance(eos_val, np.ndarray):
            seq.append(eos_val.copy())
        else:
            seq.append(eos_val)
        msk.append(eos_msk)
    # pad the sequence if necessary
    for i in range(num_padding):
        if isinstance(pad_val, np.ndarray):
            seq.append(pad_val.copy())
        else:
            seq.append(pad_val)
        msk.append(0.0)
    # the outputs are float arrays
    seq = np.array(seq)
    msk = np.array(msk).astype(np.float32)
    return seq, msk


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def indices2onehots(indices, out_dim):
    bsize, slen = indices.size()
    inds = indices.view(bsize, slen, 1)
    onehots = torch.zeros(bsize, slen, out_dim).float()
    onehots.scatter_(-1, inds, 1.0)
    return onehots.float()


def normalize_image(input_img, mean=None, std=None):
    if (mean is None) or (std is None):
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    # [0, 255] --> [0, 1]
    img_np = input_img.astype(np.float32) / 255.0
    # BGR --> RGB
    img_np = img_np[:, :, ::-1].copy()
    # Normalize
    img_np = (img_np - mean) / std
    # H x W x C --> C x H x W
    img_np = img_np.transpose((2, 0, 1))

    return img_np


def unnormalize_image(input_img, mean=None, std=None):
    if (mean is None) or (std is None):
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    # C x H x W --> H x W x C
    img_np = input_img.transpose((1, 2, 0))
    # Unnormalize
    img_np = img_np * std + mean
    # RGB --> BGR
    img_np = img_np[:, :, ::-1].copy()
    # [0, 1] --> [0, 255]
    img_np = (255.0 * img_np).astype(np.int)
    img_np = np.maximum(0, img_np)
    img_np = np.minimum(255, img_np)
    img_np = img_np.astype(np.uint8)

    return img_np


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
    X = torch.div(X, norm + 1e-10)
    return X


def l1norm(X):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=-1, keepdim=True)
    X = torch.div(torch.abs(X), norm + 1e-10)
    return X


def reduce_similarities(sims, msks, mode):
    if mode == 'max':
        sims = torch.max(sims * msks, -1)[0]
    elif mode == 'mean':
        sims = torch.sum(sims * msks, -1) / (torch.sum(msks, -1))
    return sims


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())  # (batch_size, batch_size)


def normalize_xywh(xywh, width, height):
    max_dim = max(width, height)

    # move the bounding box to the left bottom position
    offset_x = 0  # int(0.5 * (max_dim - width))
    offset_y = max_dim - height  # int(0.5 * (max_dim - height))
    cx, cy, nw, nh = xywh
    cx += offset_x;
    cy += offset_y

    # normalize the bounding box
    normalized_xywh = np.array([cx, cy, nw, nh], dtype=np.float32) / max_dim
    return normalized_xywh


def normalize_xywhs(xywhs, width, height):
    max_dim = max(width, height)

    # move the bounding boxes to the left bottom position
    offset_x = 0  # int(0.5 * (max_dim - width))
    offset_y = max_dim - height  # int(0.5 * (max_dim - height))
    normalized_xywhs = xywhs.copy()
    normalized_xywhs[:, 0] = normalized_xywhs[:, 0] + offset_x
    normalized_xywhs[:, 1] = normalized_xywhs[:, 1] + offset_y

    # normalize the bounding boxes
    normalized_xywhs = normalized_xywhs / float(max_dim)
    return normalized_xywhs


def clip_xyxy(box, width, height):
    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(box[2], width - 1)
    box[3] = min(box[3], height - 1)
    return box.astype(np.int32)


def clip_xyxys(boxes, width, height):
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], width - 1)
    boxes[:, 3] = np.minimum(boxes[:, 3], height - 1)
    return boxes.astype(np.int32)


def xywh_to_xyxy(box, width, height):
    x = box[0];
    y = box[1]
    w = box[2];
    h = box[3]

    xmin = x - 0.5 * w + 1
    xmax = x + 0.5 * w
    ymin = y - 0.5 * h + 1
    ymax = y + 0.5 * h

    xyxy = np.array([xmin, ymin, xmax, ymax])

    return clip_xyxy(xyxy, width, height)


def xywhs_to_xyxys(boxes, width, height):
    x = boxes[:, 0];
    y = boxes[:, 1]
    w = boxes[:, 2];
    h = boxes[:, 3]

    xmin = x - 0.5 * w + 1.0
    xmax = x + 0.5 * w
    ymin = y - 0.5 * h + 1.0
    ymax = y + 0.5 * h
    xyxy = np.vstack((xmin, ymin, xmax, ymax)).transpose()

    return clip_xyxys(xyxy, width, height)


def normalized_xywhs_to_xyxys(boxes):
    x = boxes[:, 0];
    y = boxes[:, 1]
    w = boxes[:, 2];
    h = boxes[:, 3]

    xmin = x - 0.5 * w
    xmax = x + 0.5 * w
    ymin = y - 0.5 * h
    ymax = y + 0.5 * h
    xyxy = np.vstack((xmin, ymin, xmax, ymax)).transpose()

    xyxy[:, 0] = np.maximum(xyxy[:, 0], 0.0)
    xyxy[:, 1] = np.maximum(xyxy[:, 1], 0.0)
    xyxy[:, 2] = np.minimum(xyxy[:, 2], 1.0)
    xyxy[:, 3] = np.minimum(xyxy[:, 3], 1.0)

    return xyxy


def xyxy_to_xywh(box):
    x = 0.5 * (box[0] + box[2])
    y = 0.5 * (box[1] + box[3])
    w = box[2] - box[0] + 1.0
    h = box[3] - box[1] + 1.0

    return np.array([x, y, w, h]).astype(np.int32)


def xyxys_to_xywhs(boxes):
    x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    y = 0.5 * (boxes[:, 1] + boxes[:, 3])
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0

    return np.vstack((x, y, w, h)).transpose()

def count_var(model):
    return sum([np.prod(p.shape) for p in model.parameters()])

def discount_cumsum(x, discount):
    """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
             x1,
             x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
        """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def filter_actions(actions, logits):
    """
    filter actions that do not show in logits
    :param actions: ndarray
    :param logit: ndarray
    :return:
    """
    actions, logits = set(list(actions)), set(list(logits))
    unfiltered_actions = actions & logits
    filtered_actions = actions - unfiltered_actions
    return list(unfiltered_actions), list(filtered_actions)

def aug_txt_with_attr(captions, attrs):
    """
    augment captions with attributes
    :param captions: list of captions
    :param attrs: list of attributes
    :return: list of captions
    """
    for idx, cap in enumerate(captions):
        start = random.randint(0, len(attrs))
        end = random.randint(start, len(attrs))
        for attr in attrs[start:end]:
            captions[idx] += ' ' + attr

    return captions

