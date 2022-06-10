

import numpy as np

def im2double(im):
    """
    Transfer np.uint8 to float type
    :param im:
    :return: output image
    """

    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out