import os
from glob import glob
import random
import numpy as np
import chainer
import skimage.io as io

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
    out = (image.astype(np.float32) - min_int) / (max_int - min_int)

    return out

def crop_pair_2d(
        image1,
        image2=None,
        crop_size=(640, 640),       # (y, x)
        coordinate=(1840, 700),     # (y, x)
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

    if aug_flag:
        x_var = int(round((0.5 - random.random()) * 10))
        y_var = int(round((0.5 - random.random()) * 10))
    else:
        x_var, y_var = 0, 0

    # get cropping position (image)
    top = coordinate[0] + y_var if x_len > crop_size[0] else 0
    left = coordinate[1] + x_var if y_len > crop_size[1] else 0
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # crop {image_A}
    cropped_image1 = image1[:, top:bottom, left:right]

    return cropped_image1


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
        img_a = min_max_normalize_one_image(crop_pair_2d(
            np.rot90(io.imread(os.path.join(self._root_path, self.split_list[i], 'a.jpeg'))).transpose(2, 0, 1)
            , crop_size=self.crop_size, coordinate=self.coordinate, aug_flag=self.train))
        img_b = min_max_normalize_one_image(crop_pair_2d(
            np.rot90(io.imread(os.path.join(self._root_path, self.split_list[i], 'b.jpeg'))).transpose(2, 0, 1)
            , crop_size=self.crop_size, coordinate=self.coordinate, aug_flag=self.train))
        return np.concatenate([img_a, img_b])

    def _get_label(self, i):
        #label = int(self.split_list[i][:self.split_list[i].find('_')]) - 1
        label = int(self.split_list[i][:self.split_list[i].find('_')])
        return label

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        x, y = self._get_image(i), self._get_label(i)
        return x.astype(np.float32), y
