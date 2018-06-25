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
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
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


def load_examples():
    if args.input_dir is None or not os.path.exists(args.input_dir):
        raise Exception("input_dir does not exist!")

    input_paths = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(args.input_dir, "*.png"))
        decode = tf.image.decode_png

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

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=args.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1]  # [height, width, channels]

        if args.multiple_A:
            # for concat features
            a_images_edge = preprocess(raw_input[:, :width // 3, :])
            a_images = preprocess(raw_input[:, width // 3:(2 * width) // 3, :])
            a_images = tf.concat(values=[a_images_edge, a_images], axis=2)
            # print('\na_images.shape: {}\n'.format(a_images.shape.as_list()))

            b_images = preprocess(raw_input[:, (2 * width) // 3:, :])
        else:
            a_images = preprocess(raw_input[:, :width // 2, :])
            b_images = preprocess(raw_input[:, width // 2:, :])

    if args.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif args.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if args.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [args.scale_size, args.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(
            tf.floor(tf.random_uniform([2], 0, args.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if args.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif args.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images],
                                                              batch_size=args.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / args.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
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
    # with tf.name_scope("global_step_summary"):
    #     tf.summary.scalar("global_step", global_step)

    with tf.name_scope('lr_decay'):
        learning_rate = tf.train.polynomial_decay(
            learning_rate=args.initial_lr,
            global_step=global_step,
            decay_steps=max_steps,
            end_learning_rate=args.end_lr
        )

    # with tf.name_scope("lr_summary"):
    #     tf.summary.scalar("lr", learning_rate)

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("d_net")]
        discrim_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("g_net")]
        gen_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1, beta2=args.beta2)
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

    examples = load_examples()
    print("examples count = %d" % examples.count)

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

    def convert(image):
        if args.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * args.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        if args.multiple_A:
            # channels = converted_inputs.shape.as_list()[3]
            converted_inputs = tf.split(converted_inputs, 2, 3)[1]
            # print('\n----642----: {}\n'.format(converted_inputs.shape.as_list()))

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

        for var in tf.global_variables():
            if var not in tf.trainable_variables():
                print(var)

        # print('\n----tf.trainable_variables()----')
        # for var in tf.trainable_variables():
        #     print(var.name)/home/thinkpad/Downloads/DT/GAN_Lib_Tensorflow/Pix2Pix/model.py

        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    # summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=10)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # summary_writer = tf.summary.FileWriter(args.output_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        print("parameter_count =", sess.run(parameter_count))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if args.checkpoint_dir is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
            saver.restore(sess, checkpoint)

        if args.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                for i in range(args.n_dis):
                    sess.run(modelNamedtuple.d_train)

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

                if should(args.display_freq):
                    fetches["display"] = display_fetches

                # results = sess.run(fetches, options=options, run_metadata=run_metadata)
                results = sess.run(fetches)

                # if should(args.summary_freq):
                #     # print("recording summary")
                #     summary_writer.add_summary(results["summary"], results["global_step"])

                if should(args.display_freq):
                    # print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(args.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * args.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * args.batch_size / rate

                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                        train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(args.save_freq):
                    print("saving model...")
                    saver.save(sess,
                               os.path.join(args.output_dir, "model"),
                               global_step=modelNamedtuple.global_step,
                               write_meta_graph=False)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()
