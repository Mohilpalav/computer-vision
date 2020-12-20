
"""
Imports we need.
Note: You may _NOT_ add any more imports than these.
"""
import argparse
import imageio
import logging
import numpy as np
from PIL import Image

def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)


def create_gaussian_kernel(size, sigma=1.0):
    """
    Creates a 2-dimensional, size x size gaussian kernel.
    It is normalized such that the sum over all values = 1. 

    Args:
        size (int):     The dimensionality of the kernel. It should be odd.
        sigma (float):  The sigma value to use 

    Returns:
        A size x size floating point ndarray whose values are sampled from the multivariate gaussian.

    See:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/eqns/eqngaus2.gif
    """

    # Ensure the parameter passed is odd
    if size % 2 != 1:
        raise ValueError('The size of the kernel should not be even.')
    
    xx , yy =(size,size)
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    result = np.array((kernel / np.sum(kernel)),dtype=np.float32)
    return result

def convolve_pixel(img, kernel, i, j):
    """
    Convolves the provided kernel with the image at location i,j, and returns the result.
    If the kernel stretches beyond the border of the image, it returns the original pixel.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.
        i (int):    The row location to do the convolution at.
        j (int):    The column location to process.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """

    # First let's validate the inputs are the shape we expect...
    if len(img.shape) != 2:
        raise ValueError(
            'Image argument to convolve_pixel should be one channel.')
    if len(kernel.shape) != 2:
        raise ValueError('The kernel should be two dimensional.')
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError(
            'The size of the kernel should not be even, but got shape %s' % (str(kernel.shape)))
    
    ksize = kernel.shape[0]
    isize = img.shape[0]
    
    imin = np.floor(ksize/2)
    jmin = np.floor(ksize/2)

    imax  = isize - imin - 1
    jmax = isize - imin - 1

    
    if(i >= imin and i <=imax and j >= jmin and j <=jmax):
        kernel = np.flipud(np.fliplr(kernel))
        
        temp = int(imin)
        value =img[i-temp:(i+temp)+1,j-temp:(j+temp)+1]
           
        sum = 0.0
        row = kernel.shape[0]
        column = kernel.shape[1]

        for m in range(0,row):
            for n in range(0,column):
                sum = sum + value[m,n]*kernel[m,n]
        return sum
    else:
        return(img[i,j])


def convolve(img, kernel):
    """
    Convolves the provided kernel with the provided image and returns the results.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """
    image = np.ones_like(img)

    row = img.shape[0]
    column = img.shape[1]
    
    for i in range(0,row):
        for j in range(0,column):
            image[i, j] = convolve_pixel(img,kernel, i, j)
    
    return image



def split(img):
    """
    Splits a image (a height x width x 3 ndarray) into 3 ndarrays, 1 for each channel.

    Args:
        img:    A height x width x 3 channel ndarray.

    Returns:
        A 3-tuple of the r, g, and b channels.
    """
    if img.shape[2] != 3:
        raise ValueError('The split function requires a 3-channel input image')
    
    r = np.array(img[:,:,0],dtype=np.uint8)
    g = np.array(img[:,:,1],dtype=np.uint8)
    b = np.array(img[:,:,2],dtype=np.uint8)

    
    return (r,g,b)


def merge(r, g, b):
    """
    Merges three images (height x width ndarrays) into a 3-channel color image ndarrays.

    Args:
        r:    A height x width ndarray of red pixel values.
        g:    A height x width ndarray of green pixel values.
        b:    A height x width ndarray of blue pixel values.

    Returns:
        A height x width x 3 ndarray representing the color image.
    """
    img = np.array(np.zeros((r.shape[0],r.shape[1],3)),dtype=np.uint8)
    img[:,:,0] = r
    img[:,:,1] = g
    img[:,:,2] = b
    
    return img

"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Blurs an image using an isotropic Gaussian kernel.')
    parser.add_argument('input', type=str, help='The input image file to blur')
    parser.add_argument('output', type=str, help='Where to save the result')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='The standard deviation to use for the Guassian kernel')
    parser.add_argument('--k', type=int, default=5,
                        help='The size of the kernel.')

    args = parser.parse_args()

    # first load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # Split it into three channels
    logging.info('Splitting it into 3 channels')
    (r, g, b) = split(inputImage)

    # compute the gaussian kernel
    logging.info('Computing a gaussian kernel with size %d and sigma %f' %
                 (args.k, args.sigma))
    kernel = create_gaussian_kernel(args.k, args.sigma)

    # convolve it with each input channel
    logging.info('Convolving the first channel')
    r = convolve(r, kernel)
    logging.info('Convolving the second channel')
    g = convolve(g, kernel)
    logging.info('Convolving the third channel')
    b = convolve(b, kernel)

    # merge the channels back
    logging.info('Merging results')
    resultImage = merge(r, g, b)

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, resultImage)
