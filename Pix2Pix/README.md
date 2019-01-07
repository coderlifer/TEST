# TEST


## How to run:

``` python
# cd to `Pix2Pix` folder and run command bellow

# train
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
  --n_dis=2 \
  --input_dir=/home/tellhow-iot/tem/webpagesaliency/pix2pix_data_2A/val \
  --output_dir=/data/tem/webpagesaliency/output_resize_512/tem/1180 \
  --max_epochs=400 \
  --which_direction=AtoB \
  --save_freq=1180 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=512 \
  --l1_weight=20.0 \
  --gan_weight=1.0 \
  --checkpoint_dir=/data/tem/webpagesaliency/output_resize_512/ \
  --checkpoint=/data/tem/webpagesaliency/output_resize_512/model-1180 \
  --multiple_A \
  --net_type='UNet' \
  --upsampe_method=depth_to_space
