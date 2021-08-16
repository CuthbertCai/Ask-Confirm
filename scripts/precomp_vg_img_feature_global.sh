txt_img_matching_global_pretrained_ckpt=/home/cgy/programs/DiaVL/logs/text_image_matching_global_1029_104643/snapshots/49.pkl
pretrained_ckpt=/home/cgy/programs/DiaVL/logs/text_image_matching_global_1029_104643/snapshots/49.pkl

python datasets/precomp_vg_img_feature_global.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --max_turns=10 \
       --n_epochs=50 \
       --lr=5e-4 \
       --exp_name=text_image_matching_global \
       --txt_img_matching_global_pretrained=$txt_img_matching_global_pretrained_ckpt \
       --pretrained=$pretrained_ckpt
