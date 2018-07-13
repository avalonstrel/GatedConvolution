
import numpy as np
import cv2
from PIL import Image
def random_ff_mask( name="ff_mask"):
    """Generate a random free form mask with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = (256,256,3)
    h,w,c = img_shape
    def npmask():

        mask = np.zeros((h,w))
        num_v = 10 + np.random.randint(6)#tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(4):

                angle = 0.01+np.random.randint(4.0)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10+np.random.randint(40)
                brush_w = 10+np.random.randint(3)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_w)
                start_x, start_y = end_x, end_y
        return mask*255

    mask_img = Image.fromarray(npmask().astype(np.uint8))
    mask_img.save("test.png")

random_ff_mask(name="ff_mask")
