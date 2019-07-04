import os
from glob import glob
import random
import numpy as np
import chainer
import matplotlib
import skimage.io as io
from skimage import morphology as mor
from skimage import measure
from skimage.color import rgb2gray
from scipy.misc import imread, imresize


def min_max_normalize_one_image(image):
    """
    normalize the itensity of an nd image based on the MIN-MAX normalization [-1, 1]
    inputs:
        volume: the input nd image
    outputs:
        out: the normalized nd image
    """

    max_int = image.max()
    min_int = image.min()
    out = (image - max_int) / (max_int - min_int)

    return out

def read_img(path, arr_type='tif'):
    """ read image array from path
    Args:
        path (str)          : path to directory which images are stored.
        arr_type (str)      : type of reading file {'npz','jpg','png','tif'}
    Returns:
        image (np.ndarray)  : image array
    """
    if arr_type == 'npz':
        image = np.load(path)['arr_0']
    elif arr_type in ('png', 'jpg'):
        image = imread(path, mode='L')
    elif arr_type == 'tif':
        image = io.imread(path)
    else:
        raise ValueError('invalid --input_type : {}'.format(arr_type))

    # ndim == 2
    if image.ndim >= 3:
        image = rgb2gray(image)
    # normalization
    image = image.astype(np.float32) / image.max()
    #image = min_max_normalize_one_image(image.astype(np.float32))
    return image


def crop_pair_2d(
        image1,
        image2=None,
        crop_size=(640, 640),       # (y, x)
        coordinate=(780, 1480),     # (y, x)
        aug_flag=True
):
    """ 2d {image, label} patches are cropped from array.
    Args:
        image1 (np.ndarray)         : Input 2d image array from 1st domain
        image2 (np.ndarray)         : Input 2d image array from 2nd domain
        crop_size ((int, int))      : Crop patch from array [y, x]
    Returns:
            cropped_image1 (np.ndarray)  : cropped 2d image array
            cropped_image2 (np.ndarray)  : cropped 2d label array
    """
    _, y_len, x_len = image1.shape
    assert y_len >= crop_size[0]
    assert x_len >= crop_size[1]
    cropped_images1 = []
    cropped_images2 = []

    if aug_flag:
        x_var = int(round((0.5 - random.random()) * 50))
        y_var = int(round((0.5 - random.random()) * 50))
    else:
        x_var, y_var = 0, 0

    # get cropping position (image)
    top = coordinate[0] + y_var if x_len > crop_size[0] else 0
    left = coordinate[1] + x_var if y_len > crop_size[1] else 0
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # crop {image_A, image_B}
    cropped_image1 = image1[:, top:bottom, left:right]
    cropped_image2 = image2[:, top:bottom, left:right]

    return cropped_image1, cropped_image2


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(
            self,
            path,
            split_list,
            crop_size='(640, 640)',
            coordinate='(780, 1480)',
            train=True
    ):
        """ Dataset preprocessing parameters
        Args:
            path (str)      : /path/to/root/directory/{images,labels}/are/located
            split_list (str): /path/to/{train, validation}_list.txt
        """
        self._root_path = path
        with open(split_list) as f:
            self.split_list = [line.rstrip() for line in f]
        self.crop_size = eval(crop_size)
        self.coordinate = eval(coordinate)
        self.train = train

    def __len__(self):
        return len(self.split_list)

    def _get_image(self, i):
        images = np.zeros((6, self.crop_size[0], self.crop_size[1]))
        img_a = np.rot90(io.imread(os.path.join(self._root_path, self.split_list[i], 'a.jpeg'))).transpose(2, 0, 1)
        img_b = np.rot90(io.imread(os.path.join(self._root_path, self.split_list[i], 'b.jpeg'))).transpose(2, 0, 1)
        images[0:3, :, :], images[3:6, :, :] = crop_pair_2d(img_a, img_b, crop_size=self.crop_size, coordinate=self.coordinate, aug_flag=self.train)
        for ch in range(len(images)):
            images[ch] = min_max_normalize_one_image(images[ch])
        return images

    def _get_label(self, i):
        #label = np.array([int(self.split_list[i][:2])])
        label = int(self.split_list[i][:2]) - 1
        return label

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        x, y = self._get_image(i), self._get_label(i)
        return x.astype(np.float32), y
