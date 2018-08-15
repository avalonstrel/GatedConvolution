import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
from bs4 import BeautifulSoup
from inpaint_model_gc import InpaintGCModel
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
parser.add_argument('--model', default='CA', type=str,
                    help='The model you use')

def read_bbox_shapes(filename):
    #file_path = os.path.join(self.path, "Annotations", filename)
    with open(filename, 'r') as reader:
        xml = reader.read()
    soup = BeautifulSoup(xml, 'xml')
    size = {}
    for tag in soup.size:
        if tag.string != "\n":
            size[tag.name] = int(tag.string)
    objects = soup.find_all('object')
    bndboxs = []
    for obj in objects:
        bndbox = {}
        for tag in obj.bndbox:
            if tag.string != '\n':
                bndbox[tag.name] = int(tag.string)

        bbox = [bndbox['ymin'], bndbox['xmin'], bndbox['ymax']-bndbox['ymin'], bndbox['xmax']-bndbox['xmin']]
        bndboxs.append(bbox)
    #print(bndboxs, size)
    return bndboxs, (size['height'], size['width'])


def bbox2mask(bbox, height, width, delta_h, delta_w, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """

    mask = np.zeros(( height, width, 3), np.float32)
    h = int(0.1*bbox[2])+np.random.randint(int(bbox[2]*0.2+1))
    w = int(0.1*bbox[3])+np.random.randint(int(bbox[3]*0.2)+1)
    mask[bbox[0]+h:bbox[0]+bbox[2]-h,
         bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
    return mask

def generate_data_batch(images_path, masks_path, batch_size=50, max_num = 1000):
    input_images = []
    file_names = []
    mask_names = []
    gt_images = []
    with open(images_path, 'r') as f, open(masks_path, 'r') as fb:
        file_paths = f.read().splitlines()
        bbox_paths = fb.read().splitlines()
    #print(file_paths, bbox_paths)
    for i, (file_path, bbox_path) in enumerate(zip(file_paths, bbox_paths)):
        #print(file_path, bbox_path)
        if i > max_num:
            break


        image = cv2.imread(file_path)
        #mask = cv2.imread(os.path.join(masks_path, mask_path))
        bboxs, shape = read_bbox_shapes(bbox_path)
        mask = bbox2mask(bboxs[0], shape[0], shape[1], 32, 32 )

        assert image.shape[:2] == mask.shape[:2]

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        image = cv2.resize(image, (256,256))
        mask = cv2.resize(mask, (256,256)).reshape((256,256,3))
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        mask01 = (mask>0).astype(np.float32)
        mask = mask01
        #mask01 = 1 - mask01
        gt_images.append(image)
        #image = image * (1-mask01) #+ (1-mask01) * np.ones(mask.shape)*255
        image = image * (1-mask01)
        #input_image = np.concatenate([image, mask], axis=2)
        input_images.append((image, mask))
        s = file_path.rfind("/")
        file_name = file_path[s+1:]
        file_names.append(file_name)
        mask_names.append(bbox_path)
        #gt_images.append(image)
        if i % batch_size == 0:
            yield (gt_images[(i-batch_size):i], input_images[(i-batch_size):i], file_names[(i-batch_size):i], mask_names[(i-batch_size):i])
    #input_images = np.array(input_images)
    #print('Shape of image: {}'.format(input_images.shape))

if __name__ == "__main__":
    ng.get_gpus(1)
    args = parser.parse_args()

    if args.model == "GC":
        model = InpaintGCModel()
    else:
        model = InpaintCAModel()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        if args.model == "GC":
            input_image = tf.placeholder(tf.float32, shape=[1, 256, 256, 3])
            input_mask = tf.placeholder(tf.float32, shape=[1, 256, 256, 1])
            input_guide = tf.placeholder(tf.float32, shape=[1, 256, 256, 1])
            output = model.build_server_graph(input_image, input_mask, input_guide )
        else:
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
        print(args.image, args.mask)
        for gt_imgs, input_images, file_names, mask_names in generate_data_batch(args.image, args.mask):
            #input_image = tf.constant(input_image, dtype=tf.float32)
            for gt_img, (input_img, mask), file_name in zip(gt_imgs, input_images, file_names):
                if args.model == "GC":
                    mask = mask[:,:,:,:1]
                    result = sess.run(output, feed_dict={input_image:input_img, input_mask:mask, input_guide:np.ones(mask.shape)})
                else:
                    mask = mask*255
                    input_img = input_img + mask

                    input_image_ = np.concatenate([input_img, mask], axis=2)
                    result = sess.run(output, feed_dict={input_image:input_image_})
                #print("Result shape:{}".format(result.shape))
                result = np.concatenate([gt_img[:,:,:,::-1], input_image_[:,:,:,::-1], result], axis=2)
                output_name = os.path.join(args.output, "output_"+file_name)
                #input_name = os.path.join(args.output, "input_"+file_name)
                print(output_name)
                cv2.imwrite(output_name, result[0][:, :, ::-1])
                #cv2.imwrite(input_name, input_img[0][:,:,::-1])
