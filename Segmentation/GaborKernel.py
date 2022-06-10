


from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
# import numba

# @numba.jit
def gaborkernel(bandwidth,gamma,psi,lambda0,theta):
    """
    Create a gabor kernel used for gabor filtering.
    :param bandwidth:
    :param gamma:
    :param psi:
    :param lambda0:
    :param theta:
    :return: kernel
    """


    # bw    = bandwidth, (1)
    # gamma = aspect ratio, (0.5)
    # psi   = phase shift, (0)
    # lambda0= wave length, (>=2)
    # theta = angle in rad, [0 pi)


    # bandwidth = 1
    # gamma = 1
    # psi = 0
    # lambda0 = 11
    # theta = np.pi/2


    sigma = lambda0/np.pi*np.sqrt(np.log(2)/2)*(2**bandwidth+1)/(2**bandwidth-1)
    sigma_x = sigma
    sigma_y = sigma/gamma

    sz=np.fix(8*np.maximum(sigma_y, sigma_x))
    if np.mod(sz,2)==0:
        sz=sz+1

    x, y=np.meshgrid(np.arange(-np.fix(sz/2), np.fix(sz/2)+1), np.arange(np.fix(sz/2), np.fix(-sz/2)-1 , -1))

    # Rotation
    x_theta=x*np.cos(theta)+y*np.sin(theta)
    y_theta=-x*np.sin(theta)+y*np.cos(theta)

    kernel = np.exp(-0.5*(x_theta**2/sigma_x**2+y_theta**2/sigma_y**2))*np.cos(2*np.pi/lambda0*x_theta+psi)

    return kernel

