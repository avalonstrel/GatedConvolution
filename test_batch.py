import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')

def generate_data_batch(images_path, masks_path, batch_size=50, max_num = 1000):
    input_images = []
    file_names = []
    mask_names = []
    gt_images = []
    for i, (file, mask_path) in enumerate(zip(os.listdir(images_path), os.listdir(masks_path))):
        #print(file, mask_path)
        if i > max_num:
            break
        if (file[-1] != "g" and file[-1] != "G") or (mask_path[-1] != "G" and mask_path[-1] != "g") :
            continue
        print(i)
        image = cv2.imread(os.path.join(images_path, file))
        mask = cv2.imread(os.path.join(masks_path, mask_path))

        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        mask01 = (mask>0).astype(np.float32)
        if mask_path[-1] == "g":
            mask01 = 1-mask01
        gt_images.append(image)
        image = image * mask01 + (1-mask01) * np.ones(mask.shape)*255
        mask = 255 - mask01*255
        input_image = np.concatenate([image, mask], axis=2)
        input_images.append(input_image)
        file_names.append(file)
        mask_names.append(mask_path)
        #gt_images.append(image)
        if i % batch_size == 0:
            yield (gt_images[(i-batch_size):i], np.array(input_images)[(i-batch_size):i], file_names[(i-batch_size):i], mask_names[(i-batch_size):i])
    #input_images = np.array(input_images)
    #print('Shape of image: {}'.format(input_images.shape))

if __name__ == "__main__":
    ng.get_gpus(1)
    args = parser.parse_args()

    model = InpaintCAModel()


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.placeholder(tf.float32, shape=[1, 256, 512, 3])
        output = model.build_server_graph(input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))

        sess.run(assign_ops)
        print('Model loaded.')

        for gt_imgs, input_images, file_names, mask_names in generate_data_batch(args.image, args.mask):
            print("a")
            #input_image = tf.constant(input_image, dtype=tf.float32)
            for gt_img, input_img, file_name in zip(gt_imgs, input_images, file_names):
                result = sess.run(output, feed_dict={input_image:input_img})
                #print("Result shape:{}".format(result.shape))
                result = np.concatenate([gt_img[:,:,:,::-1], input_img[:,:,:,::-1], result], axis=2)
                output_name = os.path.join(args.output, "output_"+file_name)
                #input_name = os.path.join(args.output, "input_"+file_name)
                print(output_name)
                cv2.imwrite(output_name, result[0][:, :, ::-1])
                #cv2.imwrite(input_name, input_img[0][:,:,::-1])
