
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


def build_A(pts1, pts2):
    """
    Constructs the intermediate matrix A used in the total least squares 
    computation of an homography mapping pts1 to pts2.

    Args:
        pts1:   An N-by-2 dimensional array of source points. This pts1[0,0] is x1, pts1[0,1] is y1, etc...
        pts2:   An N-by-2 dimensional array of desitination points.

    Returns:
        A 2Nx9 matrix A that we'll use to solve for h
    """
    if pts1.shape != pts2.shape:
        raise ValueError('The source points for homography computation must have the same shape (%s vs %s)' % (
            str(pts1.shape), str(pts2.shape)))
    if pts1.shape[0] < 4:
        raise ValueError('There must be at least 4 pairs of correspondences.')
    
    
    
    a, b = [], []
    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        r1 = np.asarray([x1, y1, 1., 0, 0, 0, -x1 * x2, -y1 * x2])
        r2 = np.asarray([0, 0, 0, x1, y1, 1., -x1 * y2, -y1 * y2])
        a.append(r1)
        a.append(r2)
        b.append([x2])
        b.append([y2])

    a = np.asarray(a)
    b = - np.asarray(b)
    A = np.column_stack((a, b))

    return A


def compute_H(pts1, pts2):
    """
    Computes an homography mapping one set of co-planar points (pts1)
    to another (pts2).

    Args:
        pts1:   An N-by-2 dimensional array of source points. This pts1[0,0] is x1, pts1[0,1] is y1, etc...
        pts2:   An N-by-2 dimensional array of desitination points.

    Returns:
        A 3x3 homography matrix that maps homogeneous coordinates of pts 1 to those in pts2.
    """
    A = build_A(pts1, pts2)

    AtA = np.dot(np.transpose(A),A)

    eig_vals, eig_vecs  = np.linalg.eigh(AtA)

    min_eig_val_index =np.argmin(eig_vals)

    min_eig_vec = eig_vecs[:,min_eig_val_index].reshape(3,3)

    return min_eig_vec


def bilinear_interp(image, point):
    """
    Looks up the pixel values in an image at a given point using bilinear
    interpolation. point is in the format (x, y).

    Args:
        image:      The image to sample
        point:      A tuple of floating point (x, y) values.

    Returns:
        A 3-dimensional numpy array representing the pixel value interpolated by "point".
    """
    x, y = point
    i, j = int(x), int(y)

    try:
        image[j+1][i+i]
    except Exception:
        return image[j][i]

    a = x - i
    b = y - j

    result = (1-a)*(1-b)*image[j][i] + a*(1-b)*image[j][i+1] + a*b*image[j+1][i+1] + (1-a)*b*image[j+1][i]

    return result


def apply_homography(H, points):
    """
    Applies the homography matrix H to the provided cartesian points and returns the results 
    as cartesian coordinates.

    Args:
        H:      A 3x3 floating point homography matrix.
        points: An Nx2 matrix of x,y points to apply the homography to.

    Returns:
        An Nx2 matrix of points that are the result of applying H to points.
    """

    homogenous_pts = []
    for point in points:
        point = list(point)
        point.append(1)
        homogenous_pts.append(point)


    homogenous_pts = np.asarray(homogenous_pts)
    ht = homogenous_pts.transpose()

    ha = np.dot(H, ht)

    cartesian = []
    hat = ha.transpose()

    for values in hat:
        x, y, z = values
        cartesian.append((x / z, y / z))
    cart_coordinates = np.asarray(cartesian)

    return cart_coordinates


def warp_homography(source, target_shape, Hinv):
    """
    Warp the source image into the target coordinate frame using a provided
    inverse homography transformation.

    Args:
        source:         A 3-channel image represented as a numpy array.
        target_shape:   A 3-tuple indicating the desired results height, width, and channels, respectively
        Hinv:           A homography that maps locations in the result to locations in the source image.

    Returns:
        An image of target_shape with source's type containing the source image warped by the homography.
    """
    result = np.zeros(target_shape, dtype='uint8')
    height, width, _ = target_shape

    rows = source.shape[0]
    columns = source.shape[1]

    for x in range(width):
        for y in range(height):

            h_result = apply_homography(Hinv, np.array([x,y]).reshape((1,2)))
            i, j = h_result[0]

            if j > rows or i > columns or j <= 0 or i <= 0:
                continue

            result[y][x] = bilinear_interp(source, h_result[0])
            
    return result


def rectify_image(image, source_points, target_points, crop):
    """
    Warps the input image source_points to the plane defined by target_points.

    Args:
        image:          The input image to warp.
        source_points:  The coordinates in the input image to warp from.
        target_points:  The coordinates to warp the corresponding source points to.
        crop:           If False, all pixels from the input image are shown. If true, the image is cropped to 
                        not show any black pixels.
    Returns:
        A new image containing the input image rectified to target_points.
    """


    H = compute_H(source_points, target_points)

    bounding_box = np.array([
        [0,0],
        [image.shape[1], 0],
        [0, image.shape[0]],
        [image.shape[1], image.shape[0]]
    ])

    warped_bounding_box = apply_homography(H, bounding_box)

    sorted_x = np.sort(warped_bounding_box[:, 0])
    sorted_y = np.sort(warped_bounding_box[:, 1])

    if crop:
        min_x = sorted_x[1]
        min_y = sorted_y[1]
    else:
        min_x = sorted_x[0]
        min_y = sorted_y[0]

    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    rectified_bounding_box = apply_homography(T, warped_bounding_box)

    inverseH = compute_H(rectified_bounding_box, bounding_box)

    sortedr_x = np.sort(rectified_bounding_box[:, 0])
    sortedr_y = np.sort(rectified_bounding_box[:, 1])


    if crop:
        max_x = sortedr_x[-2]
        max_y = sortedr_y[-2]
    else:
        max_x = sortedr_x[-1]
        max_y = sortedr_y[-1]

    shape = (int(np.around(max_y)), int(np.around(max_x)), 3)

    rectified_image = warp_homography(image, shape, inverseH)

    return rectified_image


def blend_with_mask(source, target, mask):
    """
    Blends the source image with the target image according to the mask.
    Pixels with value "1" are source pixels, "0" are target pixels, and
    intermediate values are interpolated linearly between the two.

    Args:
        source:     The source image.
        target:     The target image.
        mask:       The mask to use

    Returns:
        A new image representing the linear combination of the mask (and it's inverse)
        with source and target, respectively.
    """

    converted_mask = mask / 255

    result = (1 - converted_mask) * target + converted_mask * source

    result = result.astype(source.dtype)

    return result

def composite_image(source, target, source_pts, target_pts, mask):
    """
    Composites a masked planar region of the source image onto a
    corresponding planar region of the target image via homography warping.
    
    Args:
        source:     The source image to warp onto the target.
        target:     The target image that the source image will be warped to.
        source_pts: The coordinates on the source image.
        target_pts: The corresponding coordinates on the target image.
        mask:       A greyscale image representing the mast to use.
    """
    H = compute_H(target_pts, source_pts)

    warped_image = warp_homography(source, (target.shape[0], target.shape[1], 3), H)

    result = blend_with_mask(warped_image, target, mask)

    return result

def rectify(args):
    """
    The 'main' function for the rectify command.
    """

    # Loads the source points into a 4-by-2 array
    source_points = np.array(args.source).reshape(4, 2)

    # load the destination points, or select some smart default ones if None
    if args.dst == None:
        height = np.abs(
            np.max(source_points[:, 1]) - np.min(source_points[:, 1]))
        width = np.abs(
            np.max(source_points[:, 0]) - np.min(source_points[:, 0]))
        args.dst = [0.0, height, 0.0, 0.0, width, 0.0, width, height]

    target_points = np.array(args.dst).reshape(4, 2)

    # load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # Compute the rectified image
    result = rectify_image(inputImage, source_points, target_points, args.crop)

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, result)


def composite(args):
    """
    The 'main' function for the composite command.
    """

    # load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # load the target image
    logging.info('Loading target image %s' % (args.target))
    targetImage = load_image(args.target)

    # load the mask image
    logging.info('Loading mask image %s' % (args.mask))
    maskImage = load_image(args.mask)

    # If None, set the source points or sets them to the whole input image
    if args.source == None:
        (height, width, _) = inputImage.shape
        args.source = [0.0, height, 0.0, 0.0, width, 0.0, width, height]

    # Loads the source points into a 4-by-2 array
    source_points = np.array(args.source).reshape(4, 2)

    # Loads the target points into a 4-by-2 array
    target_points = np.array(args.dst).reshape(4, 2)

    # Compute the composite image
    result = composite_image(inputImage, targetImage,
                             source_points, target_points, maskImage)

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, result)


"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Warps an image by the computed homography between two rectangles.')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_rectify = subparsers.add_parser(
        'rectify', help='Rectifies an image such that the input rectangle is front-parallel.')
    parser_rectify.add_argument('input', type=str, help='The image to warp.')
    parser_rectify.add_argument('source', metavar='f', type=float, nargs=8,
                                help='A floating point value part of x1 y1 ... x4 y4')
    parser_rectify.add_argument(
        '--crop', help='If true, the result image is cropped.', action='store_true', default=False)
    parser_rectify.add_argument('--dst', metavar='x', type=float, nargs='+',
                                default=None, help='The four destination points in the output image.')
    parser_rectify.add_argument(
        'output', type=str, help='Where to save the result.')
    parser_rectify.set_defaults(func=rectify)

    parser_composite = subparsers.add_parser(
        'composite', help='Warps the input image onto the target points of the target image.')
    parser_composite.add_argument(
        'input', type=str, help='The source image to warp.')
    parser_composite.add_argument(
        'target', type=str, help='The target image to warp to.')
    parser_composite.add_argument('dst', metavar='f', type=float, nargs=8,
                                  help='A floating point value part of x1 y1 ... x4 y4 defining the box on the target image.')
    parser_composite.add_argument(
        'mask', type=str, help='A mask image the same size as the target image.')
    parser_composite.add_argument('--source', metavar='x', type=float, nargs='+',
                                  default=None, help='The four source points in the input image. If ommited, the whole image is used.')
    parser_composite.add_argument(
        'output', type=str, help='Where to save the result.')
    parser_composite.set_defaults(func=composite)

    args = parser.parse_args()
    args.func(args)
