import random
import threading
import logging
import time
import numpy as np
import cv2
import tensorflow as tf
from bs4 import BeautifulSoup
import pickle as pkl
#from pycocotools.coco import COCO
from neuralgym.data import feeding_queue_runner as queue_runner
from neuralgym.data.dataset import Dataset
from neuralgym.ops.image_ops import np_random_crop

#from pycocotools.coco import COCO
logger = logging.getLogger()
READER_LOCK = threading.Lock()


class DataMaskFromFNames(Dataset):
    """Data pipeline from list of filenames. Read th filenames and return masks from bbox or segmentation
    Args:
        fnamelists (list): A list of filenames or tuple of filenames, e.g.
            ['image_001.png', ...] or
            [('pair_image_001_0.png', 'pair_image_001_1.png'), ...].
        shapes (tuple): Shapes of data, e.g. [256, 256, 3] or
            [[256, 256, 3], [1]].
        random (bool): Read from `fnamelists` randomly (default to False).
        random_crop (bool): If random crop to the shape from raw image or
            directly resize raw images to the shape.
        dtypes (tf.Type): Data types, default to tf.float32.
        enqueue_size (int): Enqueue size for pipeline.
        enqueue_size (int): Enqueue size for pipeline.
        nthreads (int): Parallel threads for reading from data.
        return_fnames (bool): If True, data_pipeline will also return fnames
            (last tensor).
        filetype (str): Currently only support image.
    Examples:
        >>> fnames = ['img001.png', 'img002.png', ..., 'img999.png']
        >>> data = ng.data.DataFromFNames(fnames, [256, 256, 3])
        >>> images = data.data_pipeline(128)
        >>> sess = tf.Session(config=tf.ConfigProto())
        >>> tf.train.start_queue_runners(sess)
        >>> for i in range(5): sess.run(images)
    To get file lists, you can either use file::
        with open('data/images.flist') as f:
            fnames = f.read().splitlines()
    or glob::
        import glob
        fnames = glob.glob('data/*.png')
    You can also create fnames tuple::
        with open('images.flist') as f:
            image_fnames = f.read().splitlines()
        with open('segmentation_annotation.flist') as f:
            annotation_fnames = f.read().splitlines()
        fnames = list(zip(image_fnames, annatation_fnames))
    """
    # shape = [[256,256,3],[256,256,1]]
    def __init__(self, fnamelists, shapes, random=False, random_crop=False,
                 fn_preprocess=None, dtypes=tf.float32,
                 enqueue_size=32, queue_size=256, nthreads=8,
                 return_fnames=False, from_bbox=True, filetype='image'):
        self.fnamelists_ = self.process_fnamelists(fnamelists)
        #print(self.fnamelists_)
        self.file_length = len(self.fnamelists_)
        self.random = random
        self.random_crop = random_crop
        self.filetype = filetype
        if isinstance(shapes[0], list):
            self.shapes = shapes
        else:
            self.shapes = [shapes] * len(self.fnamelists_[0])
        if isinstance(dtypes, list):
            self.dtypes = dtypes
        else:
            self.dtypes = [dtypes] * len(self.fnamelists_[0])

        self.return_fnames = return_fnames
        self.batch_phs = [
            tf.placeholder(dtype, [None] + shape)
            for dtype, shape in zip(self.dtypes, self.shapes)]

        if self.return_fnames:
            self.shapes += [[]]
            self.dtypes += [tf.string]
            self.batch_phs.append(tf.placeholder(tf.string, [None]))
        self.enqueue_size = enqueue_size
        self.queue_size = queue_size
        self.nthreads = nthreads
        self.fn_preprocess = fn_preprocess
        if not random:
            self.index = 0
        super().__init__()
        self.create_queue()

    def process_fnamelists(self, fnamelist):
        if isinstance(fnamelist, list):
            if isinstance(fnamelist[0], str):
                return [(i,) for i in fnamelist]
            elif isinstance(fnamelist[0], tuple):
                return fnamelist
            else:
                raise ValueError('Type error for fnamelist.')
        else:
            raise ValueError('Type error for fnamelist.')

    def data_pipeline(self, batch_size):
        """Batch data pipeline.
        Args:
            batch_size (int): Batch size.
        Returns:
            A tensor with shape [batch_size] and self.shapes
                e.g. if self.shapes = ([256, 256, 3], [1]), then return
                [[batch_size, 256, 256, 3], [batch_size, 1]].
        """
        data = self._queue.dequeue_many(batch_size)
        return data

    def create_queue(self, shared_name=None, name=None):
        from tensorflow.python.ops import data_flow_ops, logging_ops, math_ops
        from tensorflow.python.framework import dtypes
        assert self.dtypes is not None and self.shapes is not None
        assert len(self.dtypes) == len(self.shapes)
        capacity = self.queue_size
        self._queue = data_flow_ops.FIFOQueue(
            capacity=capacity,
            dtypes=self.dtypes,
            shapes=self.shapes,
            shared_name=shared_name,
            name=name)

        enq = self._queue.enqueue_many(self.batch_phs)
        # create a queue runner
        queue_runner.add_queue_runner(queue_runner.QueueRunner(
            self._queue, [enq]*self.nthreads,
            feed_dict_op=[lambda: self.next_batch()],
            feed_dict_key=self.batch_phs))
        summary_name = 'fraction_of_%d_full' % capacity
        logging_ops.scalar_summary("queue/%s/%s" % (
            self._queue.name, summary_name), math_ops.cast(
                self._queue.size(), dtypes.float32) * (1. / capacity))

    def read_img(self, filename):
        #print(filename)
        img = cv2.imread(filename)

        if img is None:
            #logger.info('image is None, sleep this thread for 0.1s.{}'.format(filename))

            #time.sleep(0.1)
            return img, True
        if self.fn_preprocess:
            img = self.fn_preprocess(img)
        return img, False
    # Crowd Human
    def read_ch_bbox(self, path):
        aux_dict = pkl.load(open(path, 'rb'))
        bboxs = aux_dict["bbox"]
        bbox = random.choice(bboxs)
        extra = bbox['extra']
        shape = aux_dict["shape"]
        while 'ignore' in extra and extra['ignore'] == 1 and bbox['fbox'][0] < 0 and bbox['fbox'][1] < 0:
            bbox = random.choice(bboxs)
            extra = bbox['extra']
        fbox = bbox['fbox']
        return [[fbox[1],fbox[0],fbox[3],fbox[2]]], (shape[1], shape[0])

    def read_coco_bbox(self, path):
        aux_dict = pkl.load(open(path, 'rb'))
        bbox = aux_dict["bbox"]
        shape = aux_dict["shape"]
        #bbox = random.choice(bbox)
        #fbox = bbox['fbox']
        return [[int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])]], (shape[1], shape[0])

    def read_bbox_shapes(self, filename):
        if filename[-3:] == 'pkl' and 'Human' in filename:
            return self.read_ch_bbox(filename)
        elif filename[-3:] == 'pkl' and 'COCO' in filename:
            return self.read_coco_bbox(filename)

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


    def bbox2mask(self, bbox, height, width, delta_h, delta_w, name='mask'):
        """Generate mask tensor from bbox.

        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

        Returns:
            tf.Tensor: output with shape [1, H, W, 1]

        """

        mask = np.zeros(( height, width, 1), np.float32)
        h = int(0.1*bbox[2])+np.random.randint(int(bbox[2]*0.2+1))
        w = int(0.1*bbox[3])+np.random.randint(int(bbox[3]*0.2)+1)
        mask[bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask

    def next_batch(self):
        batch_data = []
        for _ in range(self.enqueue_size):
            error = True
            while error:
                error = False
                if random:
                    filenames = random.choice(self.fnamelists_)
                else:
                    with READER_LOCK:
                        filenames = self.fnamelists_[self.index]
                        self.index = (self.index + 1) % self.file_length
                imgs = []
                masks = []
                random_h = None
                random_w = None
                #print(list(filenames))
                for i in range(1):
                    #print(filenames[i])
                    img, error = self.read_img(filenames[0])
                    bboxs, shape = self.read_bbox_shapes(filenames[1])
                    mask = self.bbox2mask(bboxs[0], shape[0], shape[1], 32, 32 )

                    if self.random_crop:
                        img, random_h_, random_w_ = np_random_crop(
                            img, tuple(self.shapes[i][:-1]),
                            random_h, random_w, align=False)  # use last rand
                        mask, random_h, random_w = np_random_crop(
                            mask, tuple(self.shapes[i][:-1]),
                            random_h, random_w, align=False)  # use last rand
                    else:
                        if img is None or mask is None:
                            continue
                        img = cv2.resize(img, tuple(self.shapes[i][:-1]))
                        mask = cv2.resize(mask, tuple(self.shapes[i][:-1]))
                    #assert not np.isnan(((img>0).astype(np.int8)).reshape(self.shapes[i]))

                    mask = ((mask>0).astype(np.int8)).reshape((self.shapes[i][:-1]+[1,]))
                    if(np.max(mask) == 0):
                        print(bboxs[0], shape)
                        error = True
                    else:
                        imgs.append(img)
                        masks.append(mask)

            if self.return_fnames:
                batch_data.append(imgs + masks + list(filenames))
            else:
                batch_data.append(imgs + masks)
        return zip(*batch_data)

    def _maybe_download_and_extract(self):
        pass
