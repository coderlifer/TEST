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
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

import common as lib
import common.misc
import common.plot

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
parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=10, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=10, help="display progress every progress_freq steps")
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

args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

EPS = 1e-12
CROP_SIZE = 512  # 256, 512, 1024

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model",
                               "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, "
                               "gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, d_train, g_train, losses, "
                               "global_step")


def get_input_paths(input_dir):
    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    # decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png"))
        # decode = tf.image.decode_png
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files.")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    return input_paths


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def save_images(fetches, step=None):
    image_dir = os.path.join(args.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(args.output_dir, "index.html")
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
    def transform(image):
        # r = image
        if args.flip:
            image = tf.image.random_flip_left_right(image, seed=args.seed)

            # image = tf.image.random_flip_up_down(image, seed=seed)

            # k = np.random.choice([1, 2, 3, 4], 1, replace=False)[0]
            # image = tf.image.rot90(image=image, k=k)
            #
            # if k > 2:
            #     image = tf.image.transpose_image(image)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        image = tf.image.resize_images(image, [args.scale_size, args.scale_size], method=tf.image.ResizeMethod.AREA)

        if args.scale_size > CROP_SIZE:
            offset = tf.cast(
                tf.floor(tf.random_uniform([2], 0, args.scale_size - CROP_SIZE + 1, seed=args.seed)), dtype=tf.int32)
            image = tf.image.crop_to_bounding_box(image, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif args.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")

        return image

    with tf.variable_scope("load_images"):
        imgs = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        # assertion = tf.assert_equal(tf.shape(img)[-1], 3, message="image does not have 3 channels")
        # with tf.control_dependencies([assertion]):
        #     img = tf.identity(img)

        # raw_input.set_shape([None, None, None, 3])
        # print('\nraw_input.shape: {}'.format(raw_input.shape.as_list()))  # [512, 1536, 3]

        # # break apart image pair and move to range [-1, 1]
        # height = raw_input.shape.as_list()[0]  # [height, width, channels]
        # width = raw_input.shape.as_list()[1]  # [height, width, channels]
        # channels = raw_input.shape.as_list()[2]  # [height, width, channels]

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(imgs)[2]  # [args.batch_size, height, width, channels]
        if args.multiple_A:
            tf.logging.info('\nmultiple_A is enabled!\n')
            # for concat features
            a_images_edge = preprocess(imgs[:, :, :width // 3, :])  # [1, None, None, 3]
            a_images = preprocess(imgs[:, :, width // 3:(2 * width) // 3, :])  # [1, None, None, 3]
            a_images = tf.concat(values=[a_images_edge, a_images], axis=-1)  # [1, None, None, 6]

            b_images = preprocess(imgs[:, :, (2 * width) // 3:, :])  # [1, None, None, 3]
        else:
            tf.logging.info('\nmultiple_A is not enabled!\n')
            a_images = preprocess(imgs[:, :, :width // 2, :])
            b_images = preprocess(imgs[:, :, width // 2:, :])

        if args.which_direction == "AtoB":
            inputs, targets = [a_images, b_images]
        elif args.which_direction == "BtoA":
            inputs, targets = [b_images, a_images]
        else:
            raise Exception("invalid direction")

        inputs_batch = transform(inputs)  # [1, None, None, 6]
        targets_batch = transform(targets)  # [1, None, None, 3]

    return Examples(
        paths=input_paths,
        inputs=inputs_batch,
        targets=targets_batch,
        count=1,
        steps_per_epoch=1
    )


def create_model(inputs, targets):
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
    # with tf.name_scope("global_step_summary"):
    #     tf.summary.scalar("global_step", global_step)

    # with tf.name_scope('lr_decay'):
    #     learning_rate = tf.train.polynomial_decay(
    #         learning_rate=args.initial_lr,
    #         global_step=global_step,
    #         decay_steps=max_steps,
    #         end_learning_rate=args.end_lr
    #     )
    LR = 0.0002  # 2e-4  # Initial learning rate
    # decay = 1.
    decay = tf.where(
        tf.less(global_step, 35400), tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / 47200)), 0.5)

    # with tf.name_scope("lr_summary"):
    #     tf.summary.scalar("lr", learning_rate)

    with tf.name_scope("d_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("d_net")]
        discrim_optim = tf.train.AdamOptimizer(LR * decay, beta1=args.beta1, beta2=args.beta2)
        # discrim_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("g_train"):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("g_net")]
        gen_optim = tf.train.AdamOptimizer(LR * decay, beta1=args.beta1, beta2=args.beta2)
        # gen_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2)
        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    # global_step = tf.train.get_or_create_global_step()
    # incr_global_step = tf.assign(global_step, global_step + 1)

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
    if args.input_dir is None or not os.path.exists(args.input_dir):
        raise Exception("input_dir does not exist!")

    path_list = get_input_paths(args.input_dir)
    input_cnt = len(path_list)
    steps_per_epoch = math.ceil(len(path_list) / args.batch_size)
    img_list = list()
    for input_path in path_list:
        # img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        # img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        img = plt.imread(input_path)
        img_list.append(img)
    # path_list = np.asarray(path_list)
    # img_list = np.asarray(img_list)

    if args.seed is None:
        args.seed = random.randint(0, 2 ** 31 - 1)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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

    for k, v in args._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(args.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    raw_input = tf.placeholder(shape=[args.batch_size, None, None, 3], dtype=tf.float32, name='raw_input')
    input_paths = tf.placeholder(shape=None, dtype=tf.string, name='input_paths')

    examples = load_examples(raw_input, input_paths)
    # print("examples count = %d" % examples.count)

    max_steps = 2 ** 32
    if args.max_epochs is not None:
        max_steps = steps_per_epoch * args.max_epochs
    if args.max_steps is not None:
        max_steps = args.max_steps

    # inputs and targets are [batch_size, height, width, channels]
    modelNamedtuple = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(modelNamedtuple.outputs)

    def convert(image):
        if args.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * args.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)  # [None, 512, 512, 6]
        print('\n--converted_inputs.shape--: {}\n'.format(converted_inputs.shape.as_list()))

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)  # [1, None, None, 3]

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)  # [1, None, None, 3]

    with tf.name_scope("encode_images"):
        if args.multiple_A:
            # channels = converted_inputs.shape.as_list()[3]
            converted_inputs = tf.split(converted_inputs, num_or_size_splits=2, axis=-1)[1]  # [1, None, None, 3]

        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    # with tf.name_scope("inputs_summary"):
    #     tf.summary.image("inputs", converted_inputs)

    # with tf.name_scope("targets_summary"):
    #     tf.summary.image("targets", converted_targets)
    #
    # with tf.name_scope("outputs_summary"):
    #     tf.summary.image("outputs", converted_outputs)

    # tf.summary.scalar("discriminator_loss", modelNamedtuple.discrim_loss)
    # tf.summary.scalar("generator_loss_GAN", modelNamedtuple.gen_loss_GAN)
    # tf.summary.scalar("generator_loss_L1", modelNamedtuple.gen_loss_L1)

    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name + "/values", var)

    # for grad, var in modelNamedtuple.discrim_grads_and_vars + modelNamedtuple.gen_grads_and_vars:
    #     tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        # print('\n----tf.global_variables()----')
        # for var in tf.global_variables():
        #     print(var.name)

        print('\n----not in tf.trainable_variables()----')
        for var in tf.global_variables():
            if var not in tf.trainable_variables():
                print(var.name)

        print('\n----tf.trainable_variables()----')
        for var in tf.trainable_variables():
            print(var.name)

        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    # summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=20)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # summary_writer = tf.summary.FileWriter(args.output_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        print("parameter_count =", sess.run(parameter_count))

        if args.checkpoint_dir is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
            saver.restore(sess, checkpoint)

        if args.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(steps_per_epoch, max_steps)
            for _step in range(max_steps):
                start_idx = _step * args.batch_size
                end_idx = min(input_cnt, (_step + 1) * args.batch_size)
                input_img = img_list[start_idx:end_idx]
                input_path = path_list[start_idx:end_idx]

                results = sess.run(display_fetches, feed_dict={raw_input: input_img, input_paths: input_path})
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            step = 0

            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            # training
            start = time.time()
            for _epoch in range(args.max_epochs):
                shuffle_indices = np.random.permutation(np.arange(len(img_list)))
                new_img_list = []
                new_path_list = []
                for i in shuffle_indices:
                    new_img_list.append(img_list[i])
                    new_path_list.append(path_list[i])
                img_list = new_img_list
                path_list = new_path_list

                for _step in range(steps_per_epoch):
                    start_idx = _step * args.batch_size
                    end_idx = min(input_cnt, (_step + 1) * args.batch_size)
                    input_img = img_list[start_idx:end_idx]
                    input_path = path_list[start_idx:end_idx]

                    for i in range(args.n_dis):
                        sess.run(modelNamedtuple.d_train,
                                 feed_dict={raw_input: input_img, input_paths: input_path})

                    fetches = {
                        "g_train": modelNamedtuple.g_train,
                        "losses": modelNamedtuple.losses,
                        "global_step": modelNamedtuple.global_step,
                    }

                    if should(args.progress_freq):
                        fetches["discrim_loss"] = modelNamedtuple.discrim_loss
                        fetches["gen_loss_GAN"] = modelNamedtuple.gen_loss_GAN
                        fetches["gen_loss_L1"] = modelNamedtuple.gen_loss_L1

                    # if should(args.summary_freq):
                    #     fetches["summary"] = summary_op

                    # if should(args.display_freq):
                    #     fetches["display"] = display_fetches

                    # results = sess.run(fetches, options=options, run_metadata=run_metadata)
                    results = sess.run(fetches, feed_dict={raw_input: input_img, input_paths: input_path})

                    # if should(args.summary_freq):
                    #     # print("recording summary")
                    #     summary_writer.add_summary(results["summary"], results["global_step"])

                    # if should(args.display_freq):
                    #     # print("saving display images")
                    #     filesets = save_images(results["display"], step=results["global_step"])
                    #     append_index(filesets, step=True)

                    if should(args.progress_freq):
                        # global_step will have the correct step count if we resume from a checkpoint
                        train_epoch = math.ceil(results["global_step"] / steps_per_epoch)
                        train_step = (results["global_step"] - 1) % steps_per_epoch + 1
                        rate = (step + 1) * args.batch_size / (time.time() - start)
                        remaining = (max_steps - step) * args.batch_size / rate

                        print("progress epoch %d %d, step %d %d,  image/sec %0.1f  remaining %dm" %
                              (train_epoch, _epoch+1, train_step, _step+1, rate, remaining / 60))
                        print("discrim_loss", results["discrim_loss"])
                        print("gen_loss_GAN", results["gen_loss_GAN"])
                        print("gen_loss_L1", results["gen_loss_L1"])

                        lib.plot.plot('d_loss', results["discrim_loss"])
                        lib.plot.plot('g_loss_GAN', results["gen_loss_GAN"])
                        lib.plot.plot('g_loss_L1', results["gen_loss_L1"])
                        lib.plot.flush()

                    if should(args.save_freq):
                        print("saving model...")
                        saver.save(sess, os.path.join(args.output_dir, "model"),
                                   global_step=modelNamedtuple.global_step, write_meta_graph=False)
                        # lib.plot.flush()

                    lib.plot.tick()
                    step = step + 1


if __name__ == '__main__':
    train()
