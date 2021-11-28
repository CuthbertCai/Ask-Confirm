python train/train_img_attr.py \
       --cuda \
       --num_workers=1 \
       --n_feature_dim=256 \
       --focal_loss=False \
       --n_categories=1601 \
       --n_epochs=50 \
       --lr=1e-5 \
       --exp_name=image_attribute
