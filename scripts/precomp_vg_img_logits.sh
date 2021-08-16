img_attr_pretrained_ckpt=/home/cgy/programs/DiaVL/logs/image_attribute_0907_162613/snapshots/49.pkl
pretrained_ckpt=/home/cgy/programs/DiaVL/logs/image_attribute_0907_162613/snapshots/49.pkl

python datasets/precomp_vg_img_logits.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --n_epochs=50 \
       --lr=1e-4 \
       --exp_name=image_attribute \
       --img_attr_pretrained=$img_attr_pretrained_ckpt \
       --pretrained=$pretrained_ckpt

