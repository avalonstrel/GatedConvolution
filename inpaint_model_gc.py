""" Reimplemented Inapinting Gated Convolution Model """
import logging

import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty
from neuralgym.ops.gan_ops import random_interpolates

from inpaint_ops import gated_conv, gated_deconv, gen_conv, dis_conv, gen_snconv, gen_deconv
from inpaint_ops import random_ff_mask, mask_patch
from inpaint_ops import resize_mask_like, contextual_attention


logger = logging.getLogger()


class InpaintGCModel(Model):
    def __init__(self):
        super().__init__('InpaintGCModel')

    def build_inpaint_net(self, x, mask, guide, config=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x*mask, ones_x*guide], axis=3)
        #x = tf.concat([x, ones_x*guide], axis=3)
        # two stage network
        cnum = 32
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gated_conv, gated_deconv],
                          training=training, padding=padding):
            # stage1
            x = gated_conv(x, cnum, 5, 1, name='conv1')
            x = gated_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='conv3')
            x = gated_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv5')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv6')
            mask_s = resize_mask_like(mask, x)
            x = gated_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv11')
            x = gated_conv(x, 4*cnum, 3, 1, name='conv12')
            x = gated_deconv(x, 2*cnum, name='conv13_upsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='conv14')
            x = gated_deconv(x, cnum, name='conv15_upsample')
            x = gated_conv(x, cnum//2, 3, 1, name='conv16')
            x = gated_conv(x, 3, 3, 1, activation=None, name='conv17')
            x = tf.clip_by_value(x, -1., 1.)
            x_stage1 = x
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            x = x*mask + xin*(1.-mask)
            x.set_shape(xin.get_shape().as_list())
            # conv branch
            xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            x = gated_conv(xnow, cnum, 5, 1, name='xconv1')
            x = gated_conv(x, cnum, 3, 2, name='xconv2_downsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='xconv3')
            x = gated_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
            x = gated_conv(x, 4*cnum, 3, 1, name='xconv5')
            x = gated_conv(x, 4*cnum, 3, 1, name='xconv6')
            x = gated_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
            x = gated_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
            x_hallu = x
            # attention branch
            x = gated_conv(xnow, cnum, 5, 1, name='pmconv1')
            x = gated_conv(x, cnum, 3, 2, name='pmconv2_downsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='pmconv3')
            x = gated_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv5')
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv6',
                         activation=tf.nn.relu)
            #self.test_m_shape = tf.shape(mask_s)
            #self.test_x_shape = tf.shape([x)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv9')
            x = gated_conv(x, 4*cnum, 3, 1, name='pmconv10')
            pm = x
            x = tf.concat([x_hallu, pm], axis=3)

            x = gated_conv(x, 4*cnum, 3, 1, name='allconv11')
            x = gated_conv(x, 4*cnum, 3, 1, name='allconv12')
            x = gated_deconv(x, 2*cnum, name='allconv13_upsample')
            x = gated_conv(x, 2*cnum, 3, 1, name='allconv14')
            x = gated_deconv(x, cnum, name='allconv15_upsample')
            x = gated_conv(x, cnum//2, 3, 1, name='allconv16')
            x = gated_conv(x, 3, 3, 1, activation=None, name='allconv17')
            x_stage2 = tf.clip_by_value(x, -1., 1.)
        return x_stage1, x_stage2, offset_flow


    def build_sn_pgan_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            cnum = 64
            x = gen_snconv(x, cnum, 5, 2, name='conv1', training=training)
            x = gen_snconv(x, cnum*2, 5, 2, name='conv2', training=training)
            x = gen_snconv(x, cnum*4, 5, 2, name='conv3', training=training)
            x = gen_snconv(x, cnum*4, 5, 2, name='conv4', training=training)
            x = gen_snconv(x, cnum*4, 5, 2, name='conv5', training=training)
            x = gen_snconv(x, cnum*4, 5, 2, name='conv6', training=training)
            return x



    def build_graph_with_losses(self, batch_data, batch_mask, batch_guide, config, training=True,
                                summary=False, reuse=False):
        batch_pos = batch_data / 127.5 - 1.
        # generate mask, 1 represents masked point[]

        #if batch_mask is None:
        batch_mask = random_ff_mask(config)
        batch_incomplete = batch_pos*(1.-batch_mask)
        ones_x = tf.ones_like(batch_mask)[:, :, :, 0:1]
        batch_mask = ones_x*batch_mask
        batch_guide = ones_x
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, batch_mask, batch_mask, config, reuse=reuse, training=training, padding=config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*batch_mask + batch_incomplete*(1.-batch_mask)

        # local patches
        local_patch_batch_pos = mask_patch(batch_pos, batch_mask)
        local_patch_batch_predicted = mask_patch(batch_predicted, batch_mask)
        local_patch_x1 = mask_patch(x1, batch_mask)
        local_patch_x2 = mask_patch(x2, batch_mask)
        local_patch_batch_complete = mask_patch(batch_complete, batch_mask)
        #local_patch_mask = mask_patch(mask, bbox)

        # local patch l1 loss hole+out same as partial convolution
        l1_alpha = config.COARSE_L1_ALPHA
        losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x1)) # *spatial_discounting_mask(config))
        if not config.PRETRAIN_COARSE_NETWORK:
            losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x2)) # *spatial_discounting_mask(config))
        losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x1) * (1.-batch_mask))
        if not config.PRETRAIN_COARSE_NETWORK:
            losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2) * (1.-batch_mask))
        losses['ae_loss'] /= tf.reduce_mean(1.-batch_mask)

        if summary:
            scalar_summary('losses/l1_loss', losses['l1_loss'])
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            viz_img = [batch_pos, batch_incomplete, batch_complete]
            if offset_flow is not None:
                viz_img.append(
                    resize(offset_flow, scale=4,
                           func=tf.image.resize_nearest_neighbor))
            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

        if config.GAN_WITH_MASK:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(batch_mask, [config.BATCH_SIZE*2, 1, 1, 1])], axis=3)
        if config.GAN_WITH_GUIDE:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(batch_guide, [config.BATCH_SIZE*2, 1, 1, 1])], axis=3)
        #batch_pos_, batch_complete_ = tf.split(axis, value, num_split, name=None)
        # sn-pgan with gradient penalty
        if config.GAN == 'sn_pgan':
            # sn path gan
            pos_neg = self.build_sn_pgan_discriminator(batch_pos_neg, training=training, reuse=reuse)
            pos_global, neg_global = tf.split(pos_neg, 2)

            # wgan loss
            #g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
            g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
            losses['g_loss'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global
            losses['d_loss'] = d_loss_global
            # gp

            # Random Interpolate between true and false

            interpolates_global = random_interpolates(
                                    tf.concat([batch_pos, tf.tile(batch_mask, [config.BATCH_SIZE, 1, 1, 1])], axis=3),
                                    tf.concat([batch_complete, tf.tile(batch_mask, [config.BATCH_SIZE, 1, 1, 1])], axis=3))
            dout_global = self.build_sn_pgan_discriminator(interpolates_global, reuse=True)

            # apply penalty
            penalty_global = gradients_penalty(interpolates_global, dout_global, mask=batch_mask)
            losses['gp_loss'] = config.WGAN_GP_LAMBDA * penalty_global
            losses['d_loss'] = losses['d_loss'] + losses['gp_loss']

            if summary and not config.PRETRAIN_COARSE_NETWORK:
                gradients_summary(g_loss_global, batch_predicted, name='g_loss_global')
                scalar_summary('convergence/d_loss', losses['d_loss'])
                scalar_summary('convergence/global_d_loss', d_loss_global)
                scalar_summary('gan_wgan_loss/gp_loss', losses['gp_loss'])
                scalar_summary('gan_wgan_loss/gp_penalty_global', penalty_global)

        if summary and not config.PRETRAIN_COARSE_NETWORK:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')
            gradients_summary(losses['g_loss'], x1, name='g_loss_to_x1')
            gradients_summary(losses['g_loss'], x2, name='g_loss_to_x2')
            gradients_summary(losses['l1_loss'], x1, name='l1_loss_to_x1')
            gradients_summary(losses['l1_loss'], x2, name='l1_loss_to_x2')
            gradients_summary(losses['ae_loss'], x1, name='ae_loss_to_x1')
            gradients_summary(losses['ae_loss'], x2, name='ae_loss_to_x2')
        if config.PRETRAIN_COARSE_NETWORK:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
        losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        logger.info('Set L1_LOSS_ALPHA to %f' % config.L1_LOSS_ALPHA)
        logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA)
        if config.AE_LOSS:
            losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
            logger.info('Set AE_LOSS_ALPHA to %f' % config.AE_LOSS_ALPHA)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_infer_graph(self, batch_data, batch_mask, batch_guide, config, name='val'):
        """
        validation
        """
        config.MAX_DELTA_HEIGHT = 0
        config.MAX_DELTA_WIDTH = 0
        batch_pos = batch_data / 127.5 - 1.
        batch_incomplete = batch_pos*(1.-batch_mask)
        # inpaint
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, batch_mask, batch_guide, config, reuse=True,
            training=False, padding=config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        # apply mask and reconstruct
        batch_complete = batch_predicted*batch_mask + batch_incomplete*(1.-batch_mask)
        # global image visualization
        viz_img = [batch_pos, batch_incomplete, batch_complete]
        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale=4,
                       func=tf.image.resize_nearest_neighbor))
        images_summary(
            tf.concat(viz_img, axis=2),
            name+'_raw_incomplete_complete', config.VIZ_MAX_OUT)
        return batch_complete

    def build_static_infer_graph(self, batch_data, batch_mask, batch_guide, config, name):
        """
        """
        # generate mask, 1 represents masked point
        bbox = (tf.constant(config.HEIGHT//2), tf.constant(config.WIDTH//2),
                tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        return self.build_infer_graph(batch_data, batch_mask, batch_guide, config, name)


    def build_server_graph(self, batch_data, batch_mask, batch_guide, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        # batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        #masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_data / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - batch_mask)
        # inpaint
        x1, x2, flow = self.build_inpaint_net(
            batch_incomplete, batch_mask, batch_guide, reuse=reuse, training=is_training,
            config=None)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*batch_mask + batch_incomplete*(1-batch_mask)
        return batch_complete
