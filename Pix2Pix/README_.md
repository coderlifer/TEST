# TEST


## How to run:

``` python
# train
# cd to `Pix2Pix` folder and run command bellow
# webpage, 2A
CUDA_VISIBLE_DEVICES=1 python Pix2Pix/train_1.py \
  --batch_size=1 \
  --mode='train' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --n_dis=1 \
  --input_dir=/home/tellhow-iot/tem/webpagesaliency/pix2pix_data_2A/train \
  --output_dir=/data/tem/webpagesaliency/output_resize_512 \
  --max_epochs=400 \
  --which_direction=AtoB \
  --save_freq=1180 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=572 \
  --TTUR \
  --l1_weight=20.0 \
  --gan_weight=1.0 \
  --multiple_A \
  --net_type='UNet' \
  --upsampe_method='depth_to_space'

  --val_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data_2A/val

# infer
CUDA_VISIBLE_DEVICES=1 python Pix2Pix/train_1.py \
  --batch_size=1 \
  --mode='test' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --n_dis=5 \
  --input_dir=/home/tellhow-iot/tem/webpagesaliency/pix2pix_data_2A/val \
  --output_dir=/data/tem/webpagesaliency/output_resize_512/tem/38940 \
  --max_epochs=400 \
  --which_direction=AtoB \
  --save_freq=1180 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=512 \
  --l1_weight=20.0 \
  --gan_weight=1.0 \
  --checkpoint_dir=/data/tem/webpagesaliency/output_resize_512/ \
  --checkpoint=/data/tem/webpagesaliency/output_resize_512/model-38940 \
  --multiple_A \
  --net_type='UNet' \
  --upsampe_method=depth_to_space


# ---------- attention ----------
CUDA_VISIBLE_DEVICES=1 python Pix2Pix/train.py \
  --batch_size=1 \
  --mode='train' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --n_dis=1 \
  --input_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data_2A/tmp/train \
  --output_dir=/mnt/data/ILSVRC2012/webpageSaliency/output_atten_resize_512_1 \
  --max_epochs=600 \
  --which_direction=AtoB \
  --save_freq=2360 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=572 \
  --l1_weight=100.0 \
  --gan_weight=1.0 \
  --multiple_A \
  --net_type='UNet_Attention' \
  --upsampe_method=depth_to_space

  --val_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data_2A/tmp/val

# infer
CUDA_VISIBLE_DEVICES=0 python Pix2Pix/train.py \
  --batch_size=1 \
  --mode='test' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --n_dis=1 \
  --input_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data_2A/tmp/val \
  --output_dir=/mnt/data/ILSVRC2012/webpageSaliency/output_atten_resize_512_1/output_test_512 \
  --max_epochs=600 \
  --which_direction=AtoB \
  --save_freq=2360 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=572 \
  --checkpoint_dir=/mnt/data/ILSVRC2012/webpageSaliency/output_atten_resize_512_1 \
  --l1_weight=20.0 \
  --gan_weight=1.0 \
  --multiple_A \
  --net_type='UNet_Attention' \
  --upsampe_method=depth_to_space


# ---------- VGG ----------
CUDA_VISIBLE_DEVICES=1 python Pix2Pix/train_bce.py \
  --batch_size=1 \
  --mode='train' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.001 \
  --end_lr=0.0008 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --n_dis=1 \
  --input_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data_2A/tmp/train \
  --output_dir=/mnt/data/ILSVRC2012/webpageSaliency/output_atten_resize_512_1 \
  --max_epochs=600 \
  --which_direction=AtoB \
  --save_freq=2360 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=572 \
  --l1_weight=0.5 \
  --gan_weight=1.0 \
  --multiple_A \
  --net_type='UNet_Attention' \
  --upsampe_method=depth_to_space

# infer
CUDA_VISIBLE_DEVICES=0 python Pix2Pix/train_bce.py \
  --batch_size=1 \
  --mode='test' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='Modified_MiniMax' \
  --n_dis=1 \
  --input_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data_2A/tmp/val \
  --output_dir=/mnt/data/ILSVRC2012/webpageSaliency/output_atten_resize_512_0/output_test_512 \
  --max_epochs=600 \
  --which_direction=AtoB \
  --save_freq=2360 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=572 \
  --checkpoint_dir=/mnt/data/ILSVRC2012/webpageSaliency/output_atten_resize_512_0 \
  --l1_weight=0.05 \
  --gan_weight=1.0 \
  --multiple_A \
  --net_type='VGG' \
  --upsampe_method=depth_to_space


# ---------- depth_to_space ----------
# 512
CUDA_VISIBLE_DEVICES=1 python Pix2Pix/train.py \
  --batch_size=1 \
  --mode='train' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --n_dis=5 \
  --input_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data/train \
  --output_dir=../output_train \
  --max_epochs=400 \
  --which_direction=AtoB \
  --save_freq=2360 \
  --ngf=128 \
  --ndf=128 \
  --scale_size=572 \
  --l1_weight=20.0 \
  --gan_weight=1.0 \
  --upsampe_method=depth_to_space

CUDA_VISIBLE_DEVICES=1 python Pix2Pix/train.py \
  --batch_size=1 \
  --mode='test' \
  --conv_type='conv2d' \
  --input_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data/test \
  --output_dir=../output_test_512 \
  --which_direction=AtoB \
  --scale_size=572 \
  --checkpoint_dir=../output_train \
  --upsampe_method=depth_to_space


# 1024
CUDA_VISIBLE_DEVICES=0 python Pix2Pix/train.py \
  --batch_size=1 \
  --mode='train' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --n_dis=5 \
  --input_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data/train \
  --output_dir=../output_train \
  --max_epochs=400 \
  --which_direction=AtoB \
  --save_freq=2360 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=1144 \
  --l1_weight=20.0 \
  --gan_weight=1.0 \
  --upsampe_method=resize

CUDA_VISIBLE_DEVICES=0 python Pix2Pix/train.py \
  --batch_size=1 \
  --mode='test' \
  --conv_type='conv2d' \
  --input_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data/test \
  --output_dir=../output_test_1024 \
  --which_direction=AtoB \
  --scale_size=1144 \
  --checkpoint_dir=../output_train \
  --upsampe_method=resize


# ---------- depth_to_space, 512, attention ----------
CUDA_VISIBLE_DEVICES=1 python Pix2Pix/train.py \
  --batch_size=1 \
  --mode='train' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --n_dis=5 \
  --input_dir=/home/yhx/webpageSaliency/train_data/pix2pix_data/train \
  --output_dir=../output_train \
  --max_epochs=400 \
  --which_direction=AtoB \
  --save_freq=2360 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=572 \
  --l1_weight=20.0 \
  --gan_weight=1.0 \
  --net_type='UNet_Attention' \
  --upsampe_method=resize


# facades 400, cityscapes 2795 ----------
CUDA_VISIBLE_DEVICES=0 python Pix2Pix/train.py \
  --batch_size=1 \
  --mode='train' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --n_dis=5 \
  --input_dir=/home/yhx/webpageSaliency/tmp/facades/train \
  --output_dir=/home/yhx/webpageSaliency/tmp/facades/output_train_256 \
  --max_epochs=200 \
  --which_direction=BtoA \
  --save_freq=8000 \
  --ngf=128 \
  --ndf=128 \
  --scale_size=286 \
  --l1_weight=10.0 \
  --gan_weight=1.0

CUDA_VISIBLE_DEVICES=1 python Pix2Pix/train.py \
  --batch_size=1 \
  --mode='test' \
  --conv_type='conv2d' \
  --input_dir=/home/yhx/webpageSaliency/tmp/facades/val \
  --output_dir=/home/yhx/webpageSaliency/tmp/facades/output_val_256 \
  --which_direction=BtoA \
  --scale_size=286 \
  --checkpoint_dir=/home/yhx/webpageSaliency/tmp/facades/output_train_256

```
