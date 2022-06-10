

import numpy as np
import functools
import warnings
from scipy import ndimage as ndi
from skimage import morphology


def remove_small_holes(ar, min_size=64, connectivity=1, in_place=False):
    """Remove continguous holes smaller than the specified size.
    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the connected components of interest.
    min_size : int, optional (default: 64)
        The hole component size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel.
    in_place : bool, optional (default: False)
        If `True`, remove the connected components in the input array itself.
        Otherwise, make a copy.
    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.
    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small holes within connected components removed.
    Examples
    --------
    # >>> from skimage import morphology
    # >>> a = np.array([[1, 1, 1, 1, 1, 0],
    # ...               [1, 1, 1, 0, 1, 0],
    # ...               [1, 0, 0, 1, 1, 0],
    # ...               [1, 1, 1, 1, 1, 0]], bool)
    # >>> b = morphology.remove_small_holes(a, 2)
    # >>> b
    # array([[ True,  True,  True,  True,  True, False],
    #        [ True,  True,  True,  True,  True, False],
    #        [ True, False, False,  True,  True, False],
    #        [ True,  True,  True,  True,  True, False]], dtype=bool)
    # >>> c = morphology.remove_small_holes(a, 2, connectivity=2)
    # >>> c
    # array([[ True,  True,  True,  True,  True, False],
    #        [ True,  True,  True, False,  True, False],
    #        [ True, False, False,  True,  True, False],
    #        [ True,  True,  True,  True,  True, False]], dtype=bool)
    # >>> d = morphology.remove_small_holes(a, 2, in_place=True)
    # >>> d is a
    # True
    # Notes
    # -----
    # If the array type is int, it is assumed that it contains already-labeled
    # objects. The labels are not kept in the output image (this function always
    # outputs a bool image). It is suggested that labeling is completed after
    # using this function.
    # """
    # _check_dtype_supported(ar)

    #Creates warning if image is an integer image
    # if ar.dtype != bool:
    #     warnings.warn("Any labeled images will be returned as a boolean array. "
    #                   "Did you mean to use a boolean array?", UserWarning)

    if in_place:
        out = ar
    else:
        out = ar.copy()

    #Creating the inverse of ar
    if in_place:
        out = np.logical_not(out,out)
    else:
        out = np.logical_not(out)

    #removing small objects from the inverse of ar
    out = morphology.remove_small_objects(out, min_size, connectivity, in_place)

    if in_place:
        out = np.logical_not(out,out)
    else:
        out = np.logical_not(out)

    return out