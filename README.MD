# Ask&Confirm: Active Detail Enriching for Cross-Modal Retrieval with Partial Query

[Ask&Confirm: Active Detail Enriching for Cross-Modal Retrieval with Partial Query][5]  
Guanyu Cai, Jun Zhang, Xinyang Jiang, Yifei Gong, Lianghua He, Fufu yu, Pai Peng, Xiaowei Guo, Feiyue Huang, Xing Sun. ICCV2021  

Pytorch implementation of our method for iterative text-to-image retrieval with partial queries. 

## Requirement
`pip install -r requirements.txt`

## Data
Refer [DrillDown][1] to download images, features and annotations of Visual Genome.   

Download `DrillDown/data/caches` from [DrillDown][1] and put the directory under `Ask-Confirm/data`

## Prerequisite
1. Finetune Object Detector to get objects in images  
`sh scripts/train_image_attribute.sh`     

2. Train Text-Image Retrieval Models  
`sh scripts/train_text_image_matching.sh` for S-SCAN    
`sh scripts/train_text_image_matching_global.sh` for T-CMPL  

3. Get image features and objects in images  
`sh scripts/precomp_vg_img_feature.sh` for S-SCAN  
`sh scripts/precomp_vg_img_feature_global.sh` for T-CMPL  
`sh scripts/precomp_vg_img_logits.sh`  
`sh scripts/precomp_vg_attr_feature.sh`  

4. Get word statistics
`cd datasets && python word_stats.py`  
**Please replace all paths in the scripts to your own paths.**  


## Train  
**Please replace all paths in the scripts to your own paths.**

- Train Ask&Confirm    
`sh scripts/train_ppo_coherence.sh` for Ask&Confirm with S-SCAN  
`sh scripts/train_ppo_coherence_global.sh` for Ask&Confirm with T-CMPL  

## Test  
**Please replace all paths in the scripts to your own paths.**

- Test performance of classifying objects in images  
`sh scripts/test_image_attribute.sh`  

- Test performance of text-to-image retrieval  
`sh scripts/test_text_image_matching.sh` for S-SCAN  
`sh scripts/test_text_image_matching_global.sh` for T-CMPL  

- Test performance of Ask&Confirm on iterative text-to-image retrieval  
`sh scripts/test_ppo_coherence.sh` for Ask&Confirm with S-SCAN  
`sh scripts/test_ppo_coherence_global.sh` for Ask&Confirm with T-CMPL  

- Test performance of pre-defined rules on iterative text-to-image retrieval  
`sh scripts/test_ppo_rules.sh`  

## Acknowledgment
This repo was borrowed from:  
[DrillDown][1]  
[Cross-Modal-Projection-Learning][2]  
[SCAN][3]  
[openai-baselines][4]

[1]:https://github.com/uvavision/DrillDown
[2]:https://github.com/YingZhangDUT/Cross-Modal-Projection-Learning
[3]:https://github.com/kuanghuei/SCAN
[4]:https://github.com/openai/baselines
[5]:https://arxiv.org/abs/2103.01654

## Citation
```
@inproceedings{cai2021ask,
  title={Ask\&Confirm: Active Detail Enriching for Cross-Modal Retrieval With Partial Query},
  author={Cai, Guanyu and Zhang, Jun and Jiang, Xinyang and Gong, Yifei and He, Lianghua and Yu, Fufu and Peng, Pai and Guo, Xiaowei and Huang, Feiyue and Sun, Xing},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1835--1844},
  year={2021}
}
```
