import random
import time
import numpy as np
import cv2
import tensorflow as tf
from bs4 import BeautifulSoup
from neuralgym.ops.image_ops import np_random_crop

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


def bbox2mask( bbox, height, width, delta_h, delta_w, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """

    mask = np.zeros(( height, width, 1), np.float32)
    h = int(0.1*height)+np.random.randint(int(height*0.2+1))
    w = int(0.1*width)+np.random.randint(int(width*0.2)+1)
    print(bbox[0]+h,bbox[0]+bbox[2]-h,
         bbox[1]+w,bbox[1]+bbox[3]-w)
    mask[bbox[0]+h:bbox[0]+bbox[2]-h,
         bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
    return mask
bbox, shape = read_bbox_shapes("/unsullied/sharefs/_research_detection/GeneralDetection/VOC/VOCdevkit/VOC2012/Annotations/2008_000008.xml")
mask = bbox2mask(bbox[0], shape[0], shape[1], 0.3, 0.3, name='mask')
mask = cv2.resize(mask, (256,256))
def show(mode, pil_numpy):
    print(mode, len(",".join([str(i) for i in pil_numpy.flatten() if i != 0])))
#print(show("mask", mask))
print(mask.shape)
show("mask", ((mask>0).astype(np.int8)).reshape((256,256,1)))
