#!/usr/bin/env python

import argparse, logging
import os.path as osp

import utils.utils as utils

logger =logging.getLogger()
logger.setLevel(logging.INFO)

this_dir = osp.dirname(__file__)


def str2bool(v):
    return v.lower() in ('true', '1')

def log_config(args, ca):
    filename = args.model_dir +'/' + ca + '.log'
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
    logging.info(args)


parser = argparse.ArgumentParser()

##################################################################
# To be tuned
##################################################################
parser.add_argument('--use_txt_context', type=str2bool, default=False, help='whether to incorporate query history')
parser.add_argument('--use_attr', type=str2bool, default=False, help='whether add attributes to text to retrive images')
parser.add_argument('--n_feature_dim', type=int, default=256, help='dimension of the image and language features')
parser.add_argument('--instance_dim',  type=int, default=0, help='state dimensions')
parser.add_argument('--rl_finetune',   type=int, default=0, help='reinforced mode')
parser.add_argument('--policy_mode',   type=int, default=0, help='policy mode')
parser.add_argument('--explore_mode',  type=int, default=2, help='explore mode')
parser.add_argument('--final_loss_mode', type=int, default=0)
parser.add_argument('--policy_weight', type=float, default=0.1)
parser.add_argument('--l2_norm', type=str2bool, default=True, help='whether to normalize the feature vectors')
parser.add_argument('--subspace_alignment_mode', type=int, default=0)
parser.add_argument('--loss_reduction_mode', type=int, default=1)
parser.add_argument('--sim_reduction_mode', type=str, default='mean')
parser.add_argument('--temperature_lambda', type=float, default=9)
parser.add_argument('--lse_lambda', type=float, default=20)
parser.add_argument('--prob_delta', type=float, default= 1e-4)

#################################################################
# PPO hyper-parameters
#################################################################
parser.add_argument('--ppo_gamma', type=float, default=0.99)
parser.add_argument('--ppo_lambda', type=float, default=0.95)
parser.add_argument('--ppo_clip_ratio', type=float, default=0.2)
parser.add_argument('--ppo_sup_lr', type=float, default=3e-3)
parser.add_argument('--ppo_pi_lr', type=float, default=3e-4)
parser.add_argument('--ppo_v_lr', type=float, default=1e-3)
parser.add_argument('--ppo_train_pi_iters', type=int, default=80)
parser.add_argument('--ppo_train_v_iters', type=int, default=80)
parser.add_argument('--ppo_train_sup_iters', type=int, default=80)
parser.add_argument('--ppo_target_kl', type=float, default=0.01)
parser.add_argument('--ppo_epochs', type=int, default=50)
parser.add_argument('--ppo_save_freq', type=int, default=5)
parser.add_argument('--ppo_pretrained', type=str, default=None)
parser.add_argument('--ppo_update_steps', type=int, default=1000, help='must be a multiple of max_turns')
parser.add_argument('--ppo_coef_ratio', type=float, default=1.0)
parser.add_argument('--ppo_coef_ent', type=float, default=1e-4)
parser.add_argument('--ppo_coef_logit', type=float, default=1e-1)
parser.add_argument('--ppo_coef_value', type=float, default=0.5)
parser.add_argument('--ppo_coef_refine', type=float, default=1.0)
parser.add_argument('--ppo_batchsize', type=int, default=1)
parser.add_argument('--ppo_stop_rank', type=int, default=10)
parser.add_argument('--ppo_num_actions', type=int, default=1)
parser.add_argument('--ppo_mask_weight', type=float, default=0.1)
parser.add_argument('--ppo_bonus1', type=float, default=1.0)
parser.add_argument('--ppo_bonus2', type=float, default=1.0)
parser.add_argument('--ppo_rule', type=str, default='attr_freq', help='[attr_freq, query_attr_sim, coherence')
parser.add_argument('--test_turns', type=int, default=1)
parser.add_argument('--train_turns', type=int, default=1)

parser.add_argument('--negation', type=int, default=0)
parser.add_argument('--tirg_rnn', type=str2bool, default=True)
parser.add_argument('--use_soft_ctx_encoder', type=str2bool, default=False)
parser.add_argument('--paragraph_model', type=str2bool, default=False)

parser.add_argument('--cut_off_steps', type=int, default=20)
parser.add_argument('--coco_mode', type=int, default=-1)
parser.add_argument('--cross_attn', type=str2bool, default=False)
parser.add_argument('--rank_fusion', type=str2bool, default=False)
parser.add_argument('--focal_loss', type=str2bool, default=False)


##################################################################
# Resolution
##################################################################
parser.add_argument('--color_size', default=[224, 224])
parser.add_argument('--visu_size',  default=[500, 500])
parser.add_argument('--vocab_size', type=int, default=14284)
parser.add_argument('--max_turns',  type=int, default=10)
parser.add_argument('--max_attr_turns',type=int, default=10)
parser.add_argument('--n_categories', type=int, default=1601, help='object categories from VG')
parser.add_argument('--rank_batch_size', type=int, default=1000)


##################################################################
# Data
##################################################################
parser.add_argument('--pixel_means', nargs='+', type=int, default=[103.53, 116.28, 123.675])
parser.add_argument('--min_area_size', type=float, default=0.001)


##################################################################
# Language vocabulary
##################################################################
parser.add_argument('--PAD_idx', type=int, default=0)
parser.add_argument('--SOS_idx', type=int, default=1)
parser.add_argument('--EOS_idx', type=int, default=2)
parser.add_argument('--UNK_idx', type=int, default=3)


##################################################################
# Model
##################################################################
parser.add_argument('--max_violation',   type=str2bool, default=True)
parser.add_argument('--use_img_context', type=str2bool, default=False, help='whether to use retrieved image history')
parser.add_argument('--attn_type', type=str, default='general')


##################################################################
# Text encoder
##################################################################
parser.add_argument('--bidirectional', type=str2bool, default=False)
parser.add_argument('--n_rnn_layers',  type=int, default=1)
parser.add_argument('--rnn_cell',      type=str, default='GRU')
parser.add_argument('--n_embed',       type=int, default=300, help='GloVec dimension')
parser.add_argument('--emb_dropout_p', type=float, default=0.0)
parser.add_argument('--rnn_dropout_p', type=float, default=0.0)


##################################################################
# Training parameters
##################################################################
parser.add_argument('--cuda', '-gpu', action='store_true')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--finetune', type=str2bool, default=False)
parser.add_argument('--grad_norm_clipping', type=float, default=10.0)
parser.add_argument('--log_per_steps', type=int, default=10)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_epochs',  type=int, default=300)
parser.add_argument('--margin', type=float, default=0.2)


##################################################################
# evaluation
##################################################################
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--txt_img_matching_pretrained', type=str, default=None)
parser.add_argument('--txt_img_matching_global_pretrained', type=str, default=None)
parser.add_argument('--attr_img_matching_pretrained', type=str, default=None)
parser.add_argument('--img_attr_pretrained', type=str, default=None)
##################################################################


##################################################################
# Misc
##################################################################
parser.add_argument('--exp_name', type=str, default='dialog')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--eps',  type=float, default=1e-10)
parser.add_argument('--log_dir',  type=str, default=osp.join(this_dir, '..', 'logs'))
parser.add_argument('--data_dir', type=str, default=osp.join(this_dir, '..', 'data'))
parser.add_argument('--root_dir', type=str, default=osp.join(this_dir, '..'))
parser.add_argument('--vg_img_feature', type=str, default=None)
parser.add_argument('--vg_attr_img_feature', type=str, default=None)
parser.add_argument('--vg_img_logits', type=str, default=None)
parser.add_argument('--vg_attr_feature', type=str, default=None)
##################################################################


def get_train_config():
    config, unparsed = parser.parse_known_args()
    utils.prepare_directories(config)
    log_config(config, 'train')
    return config, unparsed

def get_test_config():
    config, unparsed = parser.parse_known_args()
    utils.prepare_test_directories(config)
    log_config(config, 'test')
    return config, unparsed
