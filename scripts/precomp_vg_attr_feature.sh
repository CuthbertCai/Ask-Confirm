txt_img_matching_pretrained_ckpt=/home/cgy/programs/DiaVL/logs/text_image_matching_0825_184459/snapshots/49.pkl
pretrained_ckpt=/home/cgy/programs/DiaVL/logs/text_image_matching_0825_184459/snapshots/49.pkl

python datasets/precomp_vg_attr_feature.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --max_turns=10 \
       --n_epochs=50 \
       --lr=5e-4 \
       --exp_name=text_image_matching \
       --txt_img_matching_pretrained=$txt_img_matching_pretrained_ckpt \
       --pretrained=$pretrained_ckpt
