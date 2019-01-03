"""

"""

from common.resnet_block import *
from Pix2Pix import networks


class Pix2Pix(object):
    def __init__(self):
        """
        Args:
        """

    def get_generator(self, inputs, outputs_channels, ngf=64, conv_type='conv2d', channel_multiplier=None,
                      padding='SAME', net_type='UNet', reuse=False, upsampe_method='depth_to_space'):
        """g-net
        Args:
          inputs:
          outputs_channels:
          ngf:
          conv_type:
          channel_multiplier:
          padding:
          net_type:
          reuse:
          upsampe_method:
        Return:
        """
        with tf.variable_scope('g_net', reuse=reuse):
            if net_type == 'UNet':
                output = networks.unet_generator(inputs, outputs_channels, ngf,
                                                 conv_type=conv_type,
                                                 channel_multiplier=channel_multiplier,
                                                 padding=padding,
                                                 upsampe_method=upsampe_method)
            elif net_type == 'UNet_Attention':
                output = networks.unet_g(inputs, outputs_channels, ngf,
                                         conv_type=conv_type,
                                         channel_multiplier=channel_multiplier,
                                         padding=padding,
                                         upsampe_method=upsampe_method)
            elif net_type == 'ResNet':
                output = networks.resnet_g(inputs, outputs_channels, ngf,
                                           conv_type=conv_type,
                                           channel_multiplier=channel_multiplier,
                                           padding=padding)
            elif net_type == 'VGG':
                output = networks.vgg_generator(inputs, outputs_channels, ngf,
                                                conv_type=conv_type,
                                                channel_multiplier=channel_multiplier,
                                                padding=padding,
                                                train_mode=None,
                                                trainable=None,
                                                vgg19_npy_path='/home/yhx/vgg19.npy')
            else:
                raise NotImplementedError('Generator model name [%s] is not recognized' % net_type)

        return output

    def get_discriminator(self, inputs, targets, ndf=64, spectral_normed=True, update_collection=None,
                          conv_type='conv2d', channel_multiplier=None, padding='VALID', net_type='UNet',
                          reuse=False):
        """d-net
        Args:
          inputs: real A image.
          targets: real B image or fake image generated by G.
          ndf:
          spectral_normed:
          update_collection:
          conv_type:
          channel_multiplier:
          padding:
          net_type:
          reuse:
        Return:
        """
        with tf.variable_scope('d_net', reuse=reuse):
            if net_type == 'UNet':
                output = networks.unet_discriminator(inputs, targets, ndf, spectral_normed, update_collection,
                                                     conv_type=conv_type,
                                                     channel_multiplier=channel_multiplier,
                                                     padding=padding)

            elif net_type == 'UNet_Attention':
                output = networks.unet_d(inputs, targets, ndf, spectral_normed, update_collection,
                                         conv_type=conv_type,
                                         channel_multiplier=channel_multiplier,
                                         padding=padding)
            elif net_type == 'ResNet':
                output = networks.resnet_d(inputs, targets, ndf, spectral_normed, update_collection,
                                           conv_type=conv_type,
                                           channel_multiplier=channel_multiplier,
                                           padding=padding)
            elif net_type == 'VGG':
                output = networks.vgg_discriminator(inputs, targets, ndf, spectral_normed, update_collection,
                                                    conv_type=conv_type,
                                                    channel_multiplier=channel_multiplier,
                                                    padding=padding)
            else:
                raise NotImplementedError('Discriminator model name [%s] is not recognized' % net_type)

        return output
