"""

"""

import numpy as np
import tensorflow as tf
import imageio
import os
import glob
# import pickle
# import argparse

import common.misc

load_size = 229
outputDir = './'
mode = 'train'


# parser = argparse.ArgumentParser(description='Train script')
# parser.add_argument("--load_size", type=int, default=229, help="LOAD_SIZE")
# parser.add_argument('--outputDir', type=str, default='/media/newhd/data/ILSVRC2012/train/resized_128', help="")
# parser.add_argument('--mode', type=str, default='train', help="train, eval, infer")
# args = parser.parse_args()


def load_data(fname, skip_header=0, delimiter=','):
    """Load image names and corresponding labels.

    Args:
      fname:
      skip_header:
      delimiter:

    Returns:
    """

    data = np.genfromtxt(fname, dtype=str, comments=None, delimiter=delimiter, skip_header=skip_header)

    pathes = data[:, 0]
    labels = data[:, 1]

    return pathes, labels


def load_filenames():
    filepathes = glob.glob('./*.jpg')

    np.savetxt('../filenames.txt', filepathes, fmt="%s", comments=None)

    print('Load filenames: (%d)' % len(filepathes))  # 1281167

    return filepathes


def save_data_list(input_dir, filepathes):
    """
      Read, resize and save images listed in filepathes.
    """

    cnt = 0
    bad_img = list()
    for filepath in filepathes:
        image_path = os.path.join(input_dir, filepath)
        img, path = common.misc.get_image(image_path, load_size, is_crop=False)
        if img is None:
            bad_img.append(path)
            np.savetxt('../bad_img.txt', bad_img, fmt='%s', comments=None)
            continue
        img = img.astype('uint8')

        output_file = os.path.join(outputDir, filepath)
        if not os.path.exists(os.path.dirname(output_file)):
            os.mkdir(os.path.dirname(output_file))
        imageio.imwrite(output_file, img)

        cnt += 1
        if cnt % 1000 == 0:
            print('Resizing %d / %d' % (cnt, len(filepathes)))


def resize_ILSVRC2012_dataset(input_dir):
    # For train data
    # train_ = os.path.join(root_dir, 'caffe_ilsvrc12/train.txt')
    # train_filepathes, train_labels = load_data(train_)
    train_filepathes = load_filenames()
    save_data_list(input_dir, train_filepathes)

    # # For val data
    # val_ = os.path.join(input_dir, 'caffe_ilsvrc12/val.txt')
    # val_filepathes, test_labels = load_data(val_)
    # save_data_list(input_dir, val_filepathes)


def check_files():
    """

    Args:

    Returns:
    """
    img_pathes, img_labels = load_data('../train.csv')
    # img_labels = img_labels.astype(np.int32)
    print('len(img_pathes): {}'.format(len(img_pathes)))
    print('img_pathes[0]: {}'.format(img_pathes[0]))
    print('len(img_labels): {}'.format(len(img_labels)))
    print('img_labels[0]: {}'.format(img_labels[0]))
    img_dict = dict(zip(img_pathes, img_labels))

    img_glob = glob.glob('*.jpg')
    train_data = dict()
    names_lost = list()
    for path in img_glob:
        if img_dict.get(path, -1) == -1:
            names_lost.append(path)
        else:
            train_data[path] = img_dict.get(path, -1)

    print('len(names_lost): {}'.format(len(names_lost)))
    np.savetxt('../names_lost.txt', names_lost, fmt='%s', comments=None)

    # print('len(train_data): {}'.format(len(train_data)))
    # with open('../train.pkl', 'wb') as f:
    #     pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)


# ###################  tf.data.Dataset  ################### #
def get_fnames_labels(root_dir, fpath, one_hot=False, skip_header=0, delimiter=','):
    """Load image names and corresponding labels.

    Args:
      root_dir:
      fpath: train.csv
      skip_header:
      delimiter:
      one_hot: if one_hot label

    Returns:
    """
    data = np.genfromtxt(fpath, dtype=str, comments=None, delimiter=delimiter, skip_header=skip_header)
    fnames = data[:, 0]
    fnames = np.asarray(fnames).astype(np.str)
    labels = data[:, 1]
    labels = np.asarray(labels).astype(np.str)
    # fnames_labels = dict(zip(img_pathes, img_labels))

    if one_hot:
        distinct_labels = set(labels)
        labels_name_int = dict([(b, a) for a, b in enumerate(distinct_labels)])

        labels_ = [labels_name_int[label] for label in labels]
    else:
        labels_ = labels
    labels_ = np.asarray(labels_).astype(np.int32)

    fnames_ = [os.path.join(root_dir, fname) for fname in fnames]
    fnames_ = np.asarray(fnames_).astype(np.str)

    # shuffle
    shuffle_indices = np.random.permutation(np.arange(len(fnames_)))
    fnames_ = fnames_[shuffle_indices]
    labels_ = labels_[shuffle_indices]

    print('len(fnames_): {}'.format(len(fnames_)))
    print('fnames_: {}'.format(fnames_[:2]))
    print('len(labels_): {}'.format(len(labels_)))
    print('labels_: {}'.format(labels_[:2]))

    return fnames_, labels_


def _parse_function(filename, label):
    """map_fn used in tf.data.Dataset

    Args:
      filename:
      label:

    Returns:
    """

    raw_input = tf.io.read_file(filename=filename)

    image_decoded = tf.image.decode_jpeg(contents=raw_input, channels=3)
    # image_decoded = tf.image.decode_png(contents=raw_input, channels=3)
    # image_decoded = tf.image.decode_image(contents=raw_input)

    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    # image_decoded = tf.cast(image_decoded, tf.int32)

    image_decoded = tf.image.resize_images(images=image_decoded, size=[load_size, load_size],
                                           method=tf.image.ResizeMethod.AREA,
                                           align_corners=True)

    # image_size = image_decoded.shape.as_list()
    if mode == 'train':
        image_decoded = tf.image.resize_image_with_crop_or_pad(image_decoded, load_size + 4, load_size + 4)
        image_decoded = tf.random_crop(image_decoded, [load_size, load_size, 3])
        image_decoded = tf.image.random_flip_left_right(image_decoded)
        # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
        image_decoded = tf.image.random_brightness(image_decoded, max_delta=63. / 255.)
        image_decoded = tf.image.random_saturation(image_decoded, lower=0.5, upper=1.5)
        image_decoded = tf.image.random_contrast(image_decoded, lower=0.2, upper=1.8)
        image_decoded = tf.image.per_image_standardization(image_decoded)

    return image_decoded, label


def input_fn(filenames, labels, batch_size, num_epochs=1):
    """Read image and label using tf.data

    Args:
      filenames: file path
      labels:
      batch_size:
      num_epochs:

    Returns:
    """
    filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    dataset = dataset.map(_parse_function, num_parallel_calls=None)
    dataset = dataset.shuffle(buffer_size=10000, seed=None, reshuffle_each_iteration=True)  # big than num_train
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=batch_size)

    iterator = dataset.make_one_shot_iterator()
    next_example, next_label = iterator.get_next()

    return next_example, next_label


if __name__ == '__main__':
    # with tf.Session() as session:
    #     for i in range(epochs):
    #         session.run(iterator.initializer)
    #
    #         try:
    #             # 迭代整个数据集
    #             while True:
    #                 image_batch = session.run(batch_of_images)
    #
    #         except tf.errors.OutOfRangeError:
    #             print('End of Epoch.')

    get_fnames_labels('/home/tellhow-iot/', '/home/tellhow-iot/tem/hwi/train.csv', True, 1)
