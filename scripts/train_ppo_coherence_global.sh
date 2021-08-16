vg_img_feature_npy=/home/cgy/programs/DiaVL/data/caches/vg_train_img_feat_global.npy
vg_img_logits_npy=/home/cgy/programs/DiaVL/data/caches/vg_train_img_logits.npy
txt_img_matching_global_pretrained_ckpt=/home/cgy/programs/DiaVL/logs/text_image_matching_global_1029_104643/snapshots/49.pkl
img_attr_pretrained_ckpt=/home/cgy/programs/DiaVL/logs/image_attribute_0907_162613/snapshots/49.pkl

python train/train_ppo_coherence_global.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --instance_dim=500 \
       --max_turns=20 \
       --ppo_target_kl=0.2 \
       --ppo_train_pi_iters=40 \
       --ppo_train_v_iters=40 \
       --ppo_train_sup_iters=120 \
       --ppo_pi_lr=3e-4 \
       --ppo_v_lr=1e-3 \
       --ppo_sup_lr=3e-4 \
       --ppo_epochs=5000 \
       --ppo_num_actions=10 \
       --ppo_save_freq=100 \
       --ppo_update_steps=600 \
       --ppo_bonus1=1 \
       --ppo_bonus2=1 \
       --ppo_coef_ratio=1.0 \
       --ppo_coef_ent=0.0 \
       --ppo_coef_logit=1000 \
       --ppo_coef_value=0.5 \
       --exp_name=ppo_coherence_global \
       --vg_img_feature=$vg_img_feature_npy \
       --vg_img_logits=$vg_img_logits_npy \
       --txt_img_matching_global_pretrained=$txt_img_matching_global_pretrained_ckpt \
       --img_attr_pretrained=$img_attr_pretrained_ckpt
