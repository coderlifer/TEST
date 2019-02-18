"""
Concat webpage features to real A image.

Remove unused code in train_best.py.
"""
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import glob
import math
import collections
import random
import json
import time
import imageio

sys.path.append(os.getcwd())

import common as lib
import common.misc
# import common.plot

from Pix2Pix.model import Pix2Pix

parser = argparse.ArgumentParser(description='Train script')

parser.add_argument('--batch_size', type=int, default=1, help="number of images in batch")
parser.add_argument("--mode", type=str, default='test')
parser.add_argument('--conv_type', type=str, default='conv2d', help='conv2d, depthwise_conv2d, separable_conv2d.')
parser.add_argument('--channel_multiplier', type=int, default=0,
                    help='channel_multiplier of depthwise_conv2d/separable_conv2d.')
parser.add_argument("--initial_lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--end_lr", type=float, default=0.0001, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0., help="momentum term of adam")
parser.add_argument("--beta2", type=float, default=0.9, help="momentum term of adam")
parser.add_argument("--loss_type", type=str, default='HINGE',
                    help="HINGE, WGAN, WGAN-GP, LSGAN, CGAN, Modified_MiniMax, MiniMax")
parser.add_argument('--n_dis', type=int, default=5,
                    help='Number of discriminator update per generator update.')
parser.add_argument('--input_dir', type=str, default='./', help="path to folder containing images")
parser.add_argument('--output_dir', type=str, default='./output_train', help='Directory to output the result.')
parser.add_argument('--checkpoint_dir', type=str,
                    default='/home/tellhow-iot/pix2pix_data/output_resize_512/',
                    help='Directory to stroe checkpoints and summaries.')
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--seed", type=int)
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=4000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--aspect_ratio", type=float, default=1.0,
                    help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true",
                    help="split input image into brightness (A) and color (B)")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=512,
                    help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
# parser.set_defaults(flip=True)
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

parser.add_argument("--multiple_A", dest="multiple_A", action="store_true",
                    help="whether the input is multiple A images")
parser.add_argument('--net_type', dest="net_type", type=str, default="UNet", help='')
parser.add_argument('--upsampe_method', dest="upsampe_method", type=str, default="depth_to_space",
                    help='depth_to_space, resize')

args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

EPS = 1e-12
CROP_SIZE = 512  # 256, 512, 1024

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model",
                               "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, "
                               "gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, d_train, g_train, losses, "
                               "global_step")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def load_examples(raw_input):
    with tf.name_scope("load_images"):
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        # assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        # with tf.control_dependencies([assertion]):
        #     raw_input = tf.identity(raw_input)
        # raw_input.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        # width = tf.shape(raw_input)[1]  # [height, width, channels]

        # if args.multiple_A:
        #     # for concat features
        #     a_images_edge = preprocess(raw_input[:, :width // 3, :])
        #     a_images = preprocess(raw_input[:, width // 3:(2 * width) // 3, :])
        #     a_images = tf.concat(values=[a_images_edge, a_images], axis=2)
        #
        #     b_images = preprocess(raw_input[:, (2 * width) // 3:, :])
        # else:
        a_images = preprocess(raw_input)
        b_images = preprocess(raw_input)

    if args.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif args.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    with tf.name_scope("input_images"):
        input_images = tf.image.resize_images(
            inputs, [args.scale_size, args.scale_size], method=tf.image.ResizeMethod.AREA)

    with tf.name_scope("target_images"):
        target_images = tf.image.resize_images(
            targets, [args.scale_size, args.scale_size], method=tf.image.ResizeMethod.AREA)

    paths_batch = tf.convert_to_tensor("", tf.string)
    # paths_batch, inputs_batch, targets_batch = \
    #     tf.train.batch(["", input_images, target_images], batch_size=args.batch_size)
    # steps_per_epoch = int(math.ceil(1 / args.batch_size))

    input_images = tf.expand_dims(input_images, 0)
    print('\ninput_images.shape.as_list(): {}'.format(input_images.shape.as_list()))
    target_images = tf.expand_dims(target_images, 0)
    print('target_images.shape.as_list(): {}\n'.format(target_images.shape.as_list()))

    return Examples(
        paths=paths_batch,
        inputs=input_images,
        targets=target_images,
        count=1,
        steps_per_epoch=1,
    )


def create_model(inputs, targets, max_steps):
    model = Pix2Pix()

    out_channels = int(targets.get_shape()[-1])
    outputs = model.get_generator(inputs, out_channels, ngf=args.ngf,
                                  conv_type=args.conv_type,
                                  channel_multiplier=args.channel_multiplier,
                                  padding='SAME',
                                  net_type=args.net_type, reuse=False,
                                  upsampe_method=args.upsampe_method)

    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
    predict_real = model.get_discriminator(inputs, targets, ndf=args.ndf,
                                           spectral_normed=True,
                                           update_collection=None,
                                           conv_type=args.conv_type,
                                           channel_multiplier=args.channel_multiplier,
                                           padding='VALID',
                                           net_type=args.net_type, reuse=False)

    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
    predict_fake = model.get_discriminator(inputs, outputs, ndf=args.ndf,
                                           spectral_normed=True,
                                           update_collection=None,
                                           conv_type=args.conv_type,
                                           channel_multiplier=args.channel_multiplier,
                                           padding='VALID',
                                           net_type=args.net_type, reuse=True)

    with tf.name_scope("d_loss"):
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        discrim_loss, _ = lib.misc.get_loss(predict_real, predict_fake, loss_type=args.loss_type)

        if args.loss_type == 'WGAN-GP':
            # Gradient Penalty
            alpha = tf.random_uniform(shape=[args.batch_size, 1, 1, 1], minval=0., maxval=1.)
            differences = outputs - targets
            interpolates = targets + (alpha * differences)
            # with tf.variable_scope("discriminator", reuse=True):
            gradients = tf.gradients(
                model.get_discriminator(inputs, interpolates, ndf=args.ndf,
                                        spectral_normed=True,
                                        update_collection=None,
                                        conv_type=args.conv_type,
                                        channel_multiplier=args.channel_multiplier,
                                        padding='VALID',
                                        net_type=args.net_type, reuse=True), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]) + 1e-10)
            gradient_penalty = 10 * tf.reduce_mean(tf.square((slopes - 1.)))
            discrim_loss += gradient_penalty

    with tf.name_scope("g_loss"):
        # gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        _, gen_loss_GAN = lib.misc.get_loss(predict_real, predict_fake, loss_type=args.loss_type)
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * args.gan_weight + gen_loss_L1 * args.l1_weight

    with tf.name_scope('global_step'):
        global_step = tf.train.get_or_create_global_step()

    with tf.name_scope("d_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("d_net")]
        discrim_optim = tf.train.AdamOptimizer(0.0004, beta1=args.beta1, beta2=args.beta2)
        # discrim_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("g_train"):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("g_net")]
        gen_optim = tf.train.AdamOptimizer(0.0001, beta1=args.beta1, beta2=args.beta2)
        # gen_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2)
        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    return Model(
        outputs=outputs,
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        d_train=discrim_train,
        g_train=gen_train,
        losses=update_losses,
        global_step=global_step
    )


def _create_model(input_placeholder):
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.mode == "test" or args.mode == "export":
        if args.checkpoint_dir is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(args.checkpoint_dir, "options.json"), 'r') as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(args, key, val)
        # disable these features in test mode
        args.scale_size = CROP_SIZE
        args.flip = False

    with open(os.path.join("test_options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    examples = load_examples(input_placeholder)

    max_steps = 2 ** 32
    if args.max_epochs is not None:
        max_steps = examples.steps_per_epoch * args.max_epochs
    if args.max_steps is not None:
        max_steps = args.max_steps

    # inputs and targets are [batch_size, height, width, channels]
    modelNamedtuple = create_model(examples.inputs, examples.targets, max_steps)

    # undo colorization splitting on images that we use for display/output
    outputs = deprocess(modelNamedtuple.outputs)
    outputs = tf.squeeze(outputs)

    with tf.name_scope("convert_outputs"):
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.encode_png(converted_outputs)

    # saver = tf.train.Saver(max_to_keep=20)
    # config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # sess.run(tf.global_variables_initializer())
    #
    # if args.checkpoint_dir is not None:
    #     print("loading model from checkpoint")
    #     checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
    #     saver.restore(sess, checkpoint)

    return converted_outputs


if __name__ == '__main__':
    # 调用一次，创建模型
    input_placeholder = tf.placeholder(tf.uint8, [None, None, 3], 'input_placeholder')
    outputs = _create_model(input_placeholder)

    saver = tf.train.Saver(max_to_keep=20)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if args.checkpoint_dir is not None:
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        saver.restore(sess, checkpoint)

    # 调用多次，算saliency
    webpage = imageio.imread("./cat.png")
    saliency = sess.run(outputs, feed_dict={input_placeholder: webpage})

    with open('./a.png', 'wb') as f:
        f.write(saliency)
    # saliency = np.asarray(saliency)
    # `infer.py`.
