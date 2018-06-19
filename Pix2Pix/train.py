"""
Concat webpage features to real A image.
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
import time
import cv2

sys.path.append(os.getcwd())

import common as lib
import common.misc

from Pix2Pix.model import Pix2Pix

parser = argparse.ArgumentParser(description='Train script')

parser.add_argument('--batch_size', type=int, default=64, help="number of images in batch")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
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
parser.add_argument('--checkpoint_dir', type=str, default=None,
                    help='Directory to stroe checkpoints and summaries.')
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--seed", type=int)
parser.add_argument("--max_steps", type=int, default=None, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
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
parser.add_argument("--scale_size", type=int, default=286,
                    help="scale images to this size before cropping to 256x256")
parser.add_argument("--crop_size", type=int, default=512,
                    help="crop images to 512x512")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

parser.add_argument("--multiple_A", dest="multiple_A", action="store_true",
                    help="whether the input is multiple A images")
parser.add_argument('--net_type', dest="net_type", type=str, default="UNet", help='')
parser.add_argument('--upsampe_method', dest="upsampe_method", type=str, default="depth_to_space",
                    help='depth_to_space, resize')
parser.add_argument('--val_dir', type=str, default='./', help="path to folder containing validation images.")

args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

EPS = 1e-12

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model",
                               "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, "
                               "gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, d_train, g_train, losses, "
                               "global_step")


def preprocess(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1


def deprocess(image):
    # [-1, 1] => [0, 1]
    return (image + 1) / 2


def load_data(data_dir=None):
    if data_dir is None or not os.path.exists(data_dir):
        raise Exception("input_dir does not exist!")

    input_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(data_dir, "*.png"))

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    images = []
    for path in input_paths:
        img = cv2.imread(path)
        images.append(img)

    input_paths = np.asarray(input_paths)
    images = np.asarray(images)

    # print('\nlen(input_paths): {}'.format(len(input_paths)))
    # print('input_paths: {}'.format(input_paths))
    #
    # print('\nlen(images): {}'.format(len(images)))
    # print('images[0].dtype: {}'.format(images[0].dtype))
    # print('images[0].shape: {}'.format(images[0].shape))
    # print('images[0]: {}'.format(images[0]))

    return images, input_paths


def convert(image):
    # if args.aspect_ratio != 1.0:
    #     print('\n--------------------------args.aspect_ratio != 1.0-------------------------\n')
    #     # upscale to correct aspect ratio
    #     size = [args.crop_size, int(round(args.crop_size * args.aspect_ratio))]
    #     image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)


def save_images(output_dir, fetches, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    in_path = np.array2string(fetches["paths"])
    # print('\nin_path: {}'.format(in_path))
    # for i, in_path in enumerate(fetches["paths"]):
    name, _ = os.path.splitext(os.path.basename(in_path))
    fileset = {"name": name, "step": step}
    for kind in ["inputs", "outputs", "targets"]:
        filename = name + "-" + kind + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][0]
        # print('\ncontents.shape: {}'.format(contents.shape))
        # print('contents: {}'.format(contents))
        with open(out_path, "wb") as f:
            f.write(contents)
    filesets.append(fileset)

    return filesets


def append_index(output_dir, filesets, step=False):
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def load_examples(raw_input, input_paths):
    with tf.variable_scope("load_images"):
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        # raw_input.set_shape([None, None, None, 3])
        # print('\nraw_input.shape: {}'.format(raw_input.shape.as_list()))  # [512, 1536, 3]

        # break apart image pair and move to range [-1, 1]
        height = raw_input.shape.as_list()[0]  # [height, width, channels]
        width = raw_input.shape.as_list()[1]  # [height, width, channels]
        channels = raw_input.shape.as_list()[2]  # [height, width, channels]

        if args.multiple_A:
            # for concat features

            aab = tf.split(raw_input, 3, 1)

            a_images_edge = preprocess(aab[0])
            a_images = preprocess(aab[1])
            a_images = tf.concat(values=[a_images_edge, a_images], axis=2)
            a_images = tf.reshape(a_images, [height, width // 3, channels * 2])
            # a_images_edge = preprocess(raw_input[:, :width // 3, :])
            # a_images = preprocess(raw_input[:, width // 3:(2 * width) // 3, :])
            # a_images = tf.concat(values=[a_images_edge, a_images], axis=3)
            # print('\n237.a_images.shape: {}'.format(a_images.shape.as_list()))  # [512, 512, 6]

            b_images = preprocess(aab[2])
            b_images = tf.reshape(b_images, [height, width // 3, channels])
            # b_images = preprocess(raw_input[:, (2 * width) // 3:, :])
            # print('242.b_images.shape: {}'.format(b_images.shape.as_list()))  # [512, 512, 3]
        else:
            a_images = preprocess(raw_input[:, :width // 2, :])
            b_images = preprocess(raw_input[:, width // 2:, :])

    if args.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif args.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    with tf.variable_scope("input_images"):
        input_images = tf.expand_dims(inputs, 0)

    with tf.variable_scope("target_images"):
        target_images = tf.expand_dims(targets, 0)

    # paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images],
    #                                                           batch_size=args.batch_size)
    # steps_per_epoch = int(math.ceil(len() / args.batch_size))

    # print('\n267.input_images.shape: {}'.format(input_images.shape.as_list()))  # [1, 512, 512, 6]
    # print('target_images.shape: {}'.format(target_images.shape.as_list()))  # [1, 512, 512, 3]

    return Examples(
        paths=input_paths,
        inputs=input_images,
        targets=target_images,
        count=118,
        steps_per_epoch=118,
    )


def create_model(inputs, targets, max_steps):
    model = Pix2Pix()

    out_channels = int(targets.shape.as_list()[-1])
    outputs = model.get_generator(inputs, out_channels, ngf=args.ngf,
                                  conv_type=args.conv_type,
                                  channel_multiplier=args.channel_multiplier,
                                  padding='SAME',
                                  net_type=args.net_type,
                                  reuse=False,
                                  upsampe_method=args.upsampe_method)

    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
    predict_real = model.get_discriminator(inputs, targets, ndf=args.ndf,
                                           spectral_normed=True,
                                           update_collection=None,
                                           conv_type=args.conv_type,
                                           channel_multiplier=args.channel_multiplier,
                                           padding='VALID',
                                           net_type=args.net_type,
                                           reuse=False)

    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
    predict_fake = model.get_discriminator(inputs, outputs, ndf=args.ndf,
                                           spectral_normed=True,
                                           update_collection=None,
                                           conv_type=args.conv_type,
                                           channel_multiplier=args.channel_multiplier,
                                           padding='VALID',
                                           net_type=args.net_type,
                                           reuse=True)

    # with tf.name_scope("discriminator_loss"):
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

    # with tf.name_scope("generator_loss"):
    # gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
    _, gen_loss_GAN = lib.misc.get_loss(predict_real, predict_fake, loss_type=args.loss_type)
    gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
    gen_loss = gen_loss_GAN * args.gan_weight + gen_loss_L1 * args.l1_weight

    # with tf.variable_scope('global_step'):
    global_step = tf.train.get_or_create_global_step()

    # with tf.variable_scope('lr_decay'):
    learning_rate = tf.train.polynomial_decay(
        learning_rate=args.initial_lr,
        global_step=global_step,
        decay_steps=max_steps,
        end_learning_rate=args.end_lr
    )

    with tf.variable_scope("d_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("d_net")]
        discrim_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.variable_scope("g_train"):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("g_net")]
        gen_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2)
        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    # global_step = tf.train.get_or_create_global_step()
    # incr_global_step = tf.assign(global_step, global_step + 1)

    # print('\n----tf.global_variables()----')
    # for var in tf.global_variables():
    #     print(var.name)

    print('\n----tf.trainable_variables()----')
    for var in tf.trainable_variables():
        print(var.name)

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


def train():
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    for k, v in args._get_kwargs():
        print(k, "=", v)

    raw_input = tf.placeholder(shape=[512, 1536, 3], name='raw_input', dtype=tf.uint8)
    input_paths = tf.placeholder(shape=None, name='input_paths', dtype=tf.string)

    examples = load_examples(raw_input, input_paths)
    # print('\n406.examples.inputs.shape: {}'.format(examples.inputs.shape.as_list()))  # [1, 512, 512, 6]
    # print('examples.targets.shape: {}'.format(examples.targets.shape.as_list()))  # [1, 512, 512, 3]

    max_steps = 2 ** 32
    if args.max_epochs is not None:
        max_steps = examples.steps_per_epoch * args.max_epochs
    if args.max_steps is not None:
        max_steps = args.max_steps

    # inputs and targets are [batch_size, height, width, channels]
    modelNamedtuple = create_model(examples.inputs, examples.targets, max_steps)

    # undo colorization splitting on images that we use for display/output
    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(modelNamedtuple.outputs)
    # print('\n----422.inputs.shape----: {}'.format(inputs.shape.as_list()))  # [1, 512, 512, 6]
    # print('----targets.shape----: {}'.format(targets.shape.as_list()))  # [1, 512, 512, 3]
    # print('----outputs.shape----: {}'.format(outputs.shape.as_list()))  # [1, 512, 512, 3]

    # reverse any processing on images so they can be written to disk or displayed to user
    converted_inputs = convert(inputs)  # [1, 512, 512, 6]
    converted_targets = convert(targets)  # [1, 512, 512, 3]
    converted_outputs = convert(outputs)  # [1, 512, 512, 3]
    # print('\n----430.converted_inputs.shape----: {}\n'.format(converted_inputs.shape.as_list()))
    # print('----converted_targets.shape----: {}'.format(converted_targets.shape.as_list()))
    # print('----converted_outputs.shape----: {}'.format(converted_outputs.shape.as_list()))

    with tf.variable_scope("encode_images"):
        if args.multiple_A:
            # channels = converted_inputs.shape.as_list()[3]
            converted_inputs = tf.split(converted_inputs, 2, 3)[1]  # [1, 512, 512, 3]
            # print('\n----438.converted_inputs.shape----: {}\n'.format(converted_inputs.shape.as_list()))

        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    with tf.variable_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    train_data, train_paths = load_data(args.input_dir)
    val_data, val_paths = load_data(args.val_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print("parameter_count =", sess.run(parameter_count))

        start = time.time()
        for step in range(max_steps):
            idx = step % 118
            if idx == 0:
                shuffle_indices = np.random.permutation(np.arange(len(train_data)))
                train_data = train_data[shuffle_indices]
                train_paths = train_paths[shuffle_indices]
                # train_data_new = []
                # train_paths_new = []
                # for i in shuffle_indices:
                #     train_data_new.append(train_data[i])
                #     train_paths_new.append(train_paths[i])
                # train_data = train_data_new
                # train_paths = train_paths_new

            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            for i in range(args.n_dis):
                sess.run(modelNamedtuple.d_train,
                         feed_dict={raw_input: train_data[idx],
                                    input_paths: train_paths[idx]})

            fetches = {
                "g_train": modelNamedtuple.g_train,
                "losses": modelNamedtuple.losses,
                "global_step": modelNamedtuple.global_step,
            }

            if should(args.progress_freq):
                fetches["discrim_loss"] = modelNamedtuple.discrim_loss
                fetches["gen_loss_GAN"] = modelNamedtuple.gen_loss_GAN
                fetches["gen_loss_L1"] = modelNamedtuple.gen_loss_L1
                # fetches["examples.inputs"] = examples.inputs
                # fetches["examples.targets"] = examples.targets
                # fetches["modelNamedtuple.outputs"] = modelNamedtuple.outputs

            # results = sess.run(fetches, options=options, run_metadata=run_metadata)
            results = sess.run(fetches,
                               feed_dict={raw_input: train_data[idx],
                                          input_paths: train_paths[idx]})

            if should(args.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * args.batch_size / (time.time() - start)
                remaining = (max_steps - step) * args.batch_size / rate

                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                print("global_step", results["global_step"])
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])
                # print("examples.inputs", results["examples.inputs"])
                # print("examples.targets", results["examples.targets"])
                # print("modelNamedtuple.outputs", results["modelNamedtuple.outputs"])

            if should(args.save_freq):
                tf.logging.info('validating...')

                train_epoch_ = math.ceil(results["global_step"] / examples.steps_per_epoch)
                val_output_dir = os.path.join(args.output_dir, str(train_epoch_))
                if not os.path.exists(val_output_dir):
                    os.makedirs(val_output_dir)

                max_val_steps = len(val_data)
                for val_step in range(max_val_steps):
                    results = sess.run(display_fetches,
                                       feed_dict={raw_input: val_data[val_step],
                                                  input_paths: val_paths[val_step]})
                    filesets = save_images(val_output_dir, results)
                    for i, f in enumerate(filesets):
                        print("evaluated image", f["name"])
                    index_path = append_index(val_output_dir, filesets)
                print("wrote index at", index_path)


if __name__ == '__main__':
    train()
