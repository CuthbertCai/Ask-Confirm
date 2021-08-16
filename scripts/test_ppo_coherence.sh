vg_img_feature_npy=/home/cgy/programs/DiaVL/data/caches/vg_test_img_feat_global.npy
vg_img_logits_npy=/home/cgy/programs/DiaVL/data/caches/vg_test_img_logits.npy
txt_img_matching_pretrained_ckpt=/home/cgy/programs/DiaVL/logs/text_image_matching_0825_184459/snapshots/49.pkl
img_attr_pretrained_ckpt=/home/cgy/programs/DiaVL/logs/image_attribute_0907_162613/snapshots/49.pkl
ppo_pretrained_ckpt=/home/cgy/programs/DiaVL/logs/ppo_coherence_1028_165732/snapshots/500.pkl
pretrained_ckpt=/home/cgy/programs/DiaVL/logs/ppo_coherence_1028_165732/snapshots/500.pkl

python test/test_ppo_coherence.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --n_epochs=50 \
       --instance_dim=100 \
       --max_turns=10 \
       --test_turns=4 \
       --lr=1e-5 \
       --ppo_target_kl=0.7 \
       --ppo_num_actions=3 \
       --exp_name=ppo_coherence \
       --vg_img_feature=$vg_img_feature_npy \
       --vg_img_logits=$vg_img_logits_npy \
       --txt_img_matching_pretrained=$txt_img_matching_pretrained_ckpt \
       --img_attr_pretrained=$img_attr_pretrained_ckpt \
       --ppo_pretrained=$ppo_pretrained_ckpt \
       --pretrained=$pretrained_ckpt
