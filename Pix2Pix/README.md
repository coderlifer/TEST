# TEST

type | AUC-Judd/Borji/shuffled(C)  | CC(B) | NSS(C)
---- | ---- | --- | ---
n1_1_1| 0.8306/0.7734/0.7047 | 0.6872 | 1.5937


[n1_1_1]: IN, SN, Hinge, 512pix, noINinDiscri, 64ngf, depth_to_space, 400epoch, 2A, n_dis=1, D_lr=0.0004, G_lr=0.0001, SNinGandD, TTUR, `Resnet_G`(Use resnet architecture to extract image featrues, encode to 4*4, and decode from it, **with_att**), `UNET_D`(D's output is [N, 30, 30, 1], **without_att**), **pixel cross entropy, l1_weight=0.05**. (See BigGAN'structure.) 


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
  --content_loss='bce' \
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
  --l1_weight=0.05 \
  --gan_weight=1.0 \
  --multiple_A \
  --net_type='ResNet' \
  --upsampe_method='depth_to_space'


  --nasnet=None \
# infer
# 47200 29500 28320 27140 25960 23600
CUDA_VISIBLE_DEVICES=0 python Pix2Pix/train_1.py \
  --batch_size=1 \
  --mode='test' \
  --conv_type='conv2d' \
  --channel_multiplier=0 \
  --initial_lr=0.0002 \
  --end_lr=0.0001 \
  --beta1=0. \
  --beta2=0.9 \
  --loss_type='HINGE' \
  --g_bce \
  --n_dis=1 \
  --input_dir=/home/tellhow-iot/tem/webpagesaliency/pix2pix_data_2A/val \
  --output_dir=/data/tem/webpagesaliency/output_resize_512_/tem/1180 \
  --max_epochs=400 \
  --which_direction=AtoB \
  --save_freq=1180 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=512 \
  --TTUR \
  --l1_weight=0.05 \
  --gan_weight=1.0 \
  --checkpoint_dir=/data/tem/webpagesaliency/output_resize_512_/ \
  --checkpoint=/data/tem/webpagesaliency/output_resize_512_/model-1180 \
  --multiple_A \
  --net_type='ResNet' \
  --upsampe_method=depth_to_space


# 47200 46020 29500 28320 27140 25960 23600
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
  --content_loss='bce' \
  --n_dis=1 \
  --input_dir=/home/tellhow-iot/tem/webpagesaliency/pix2pix_data_2A/val \
  --output_dir=/data/tem/webpagesaliency/output_resize_512/tem/11800 \
  --max_epochs=400 \
  --which_direction=AtoB \
  --save_freq=1180 \
  --ngf=64 \
  --ndf=64 \
  --scale_size=512 \
  --TTUR \
  --l1_weight=0.05 \
  --gan_weight=1.0 \
  --checkpoint_dir=/data/tem/webpagesaliency/output_resize_512/ \
  --checkpoint=/data/tem/webpagesaliency/output_resize_512/model-11800 \
  --multiple_A \
  --net_type='ResNet' \
  --upsampe_method=depth_to_space
