

import numpy as np

def float2Uint(Image_float):
    """
    Transfer float image to np.uint8 type
    :param Image_float:
    :return:LnGray
    """

    MaxLn = np.max(Image_float)
    MinLn = np.min(Image_float)
    # LnGray = 255*(Image_float - MinLn)//(MaxLn - MinLn + 1e-6)
    LnGray = 255 * ((Image_float - MinLn) / float((MaxLn - MinLn + 1e-6)))
    LnGray = np.array(LnGray, dtype = np.uint8)

    return LnGray
