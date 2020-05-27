"""
Copyright (C) 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# pylint: disable=C0111,E1102,C0103,W0703,W0511,E1136
import os
import time
import glob
import shutil
from math import ceil
import random
import numpy as np
import skimage.util as skim_util
from skimage import color
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from file_utils import ls


def is_photo(fname):
    """ extensions that have been tested with so far """
    extensions = {".jpg", ".png", ".jpeg", '.tif', '.tiff'}
    fname_ext = os.path.splitext(fname)[1].lower()
    return fname_ext in extensions

def normalize_tile(tile):
    if np.min(tile) < np.max(tile):
        tile = rescale_intensity(tile, out_range=(0, 1))
    assert np.min(tile) >= 0, f"tile min {np.min(tile)}"
    assert np.max(tile) <= 1, f"tile max {np.max(tile)}"
    return tile

def load_train_image_and_annot(dataset_dir, train_annot_dir):
    max_attempts = 60
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        # file systems are unpredictable.
        # We may have problems reading the file.
        # try-catch to avoid this.
        # (just try again)
        try:
            # This might take ages, profile and optimize
            fnames = ls(train_annot_dir)
            fnames = [a for a in fnames if is_photo(a)]
            fname = random.sample(fnames, 1)[0]
            annot_path = os.path.join(train_annot_dir, fname)
            image_path_part = os.path.join(dataset_dir,
                                           os.path.splitext(fname)[0])
            # it's possible the image has a different extenstion
            # so use glob to get it
            image_path = glob.glob(image_path_part + '.*')[0]
            image = load_image(image_path)
            annot = img_as_ubyte(imread(annot_path))
            assert np.sum(annot) > 0
            assert image.shape[2] == 3 # should be RGB
            # also return fname for debugging purposes.
            return image, annot, fname
        except Exception as e:
            # This could be due to an empty annotation saved by the user.
            # Which happens rarely due to deleting all labels in an 
            # existing annotation and is not a problem.
            # give it some time and try again.
            time.sleep(0.1)
    if attempts == max_attempts:
        raise Exception('Could not load annotation and photo')


def get_class_map(annot, class_rgb):
    """ Return binary map defining locations of an image which
        are equal to class_rgb """
    assert annot.dtype == np.ubyte, (
        f'Annot dtype: {annot.dtype} but should be np.ubyte. '
        'Each channel in annotation should be 0-255 range')
    assert len(class_rgb) >= 3, str(class_rgb)
    class_r, class_g, class_b = class_rgb[:3]
    # get the specific RGB channels
    r_channel = annot[:, :, 0]
    g_channel = annot[:, :, 1]
    b_channel = annot[:, :, 2]

    # we need to get a map of all places where this class is
    # defined in the annotation. E.g all places where this
    # color exists.
    class_map = ((r_channel == class_r) *
                 (g_channel == class_g) *
                 (b_channel == class_b))
    return class_map


def annot_to_target_and_mask(annot, target_classes):
    """
    The annotation is an image where each pixel has an RGB value
    Convert this to a 2D image where each pixel is a value from 0 to maximum class index
    Defining the specific class at that location in the image.
    """
    assert annot.dtype == np.ubyte, (
        f'Annot dtype: {annot.dtype} but should be np.ubyte. '
        'Each channel in annotation should be 0-255 range')

    r_channel = annot[:, :, 0]
    g_channel = annot[:, :, 1]
    b_channel = annot[:, :, 2]

    # mask defines all places where something is defined.
    mask = (r_channel + g_channel + b_channel) > 0
    
    # target defines the class at each pixel location.
    target = np.zeros(r_channel.shape)

    # We have multiple classes.
    for i, target_class in enumerate(target_classes):
        # each class has an RGB color associated with it.
        class_r, class_g, class_b = target_class

        # we need to get a map of all places where this class is
        # defined in the annotation. E.g all places where this
        # color exists.
        class_map = ((r_channel == class_r) *
                     (g_channel == class_g) *
                     (b_channel == class_b))
        target[class_map] = i
    return target, mask


def pad(image, width: int, mode='reflect', constant_values=0):
    # only pad the first two dimensions
    pad_width = [(width, width), (width, width)]
    if len(image.shape) == 3:
        # don't pad channels
        pad_width.append((0, 0))
    if mode == 'reflect':
        return skim_util.pad(image, pad_width, mode)
    return skim_util.pad(image, pad_width, mode=mode,
                         constant_values=constant_values)


def add_salt_pepper(image, intensity):
    image = np.array(image)
    white = [1, 1, 1]
    black = [0, 0, 0]
    if len(image.shape) == 2 or image.shape[-1] == 1:
        white = 1
        black = 0
    num = np.ceil(intensity * image.size).astype(np.int)
    x_coords = np.floor(np.random.rand(num) * image.shape[1])
    x_coords = x_coords.astype(np.int)
    y_coords = np.floor(np.random.rand(num) * image.shape[0]).astype(np.int)
    image[x_coords, y_coords] = white
    x_coords = np.floor(np.random.rand(num) * image.shape[1]).astype(np.int)
    y_coords = np.floor(np.random.rand(num) * image.shape[0]).astype(np.int)
    image[y_coords, x_coords] = black
    return image

def add_gaussian_noise(image, sigma):
    assert np.min(image) >= 0, f"should be at least 0, min {np.min(image)}"
    assert np.max(image) <= 1, f"can't exceed 1, max {np.max(image)}"
    gaussian_noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    gaussian_noise = gaussian_noise.reshape(image.shape)
    return image + gaussian_noise

def get_coords(padded_im, image, in_tile_shape, out_tile_shape):

    horizontal_count = ceil(image.shape[1] / out_tile_shape[1])
    vertical_count = ceil(image.shape[0] / out_tile_shape[0])

    # first split the image based on the tiles that fit
    x_coords = [h*out_tile_shape[1] for h in range(horizontal_count-1)]
    y_coords = [v*out_tile_shape[0] for v in range(vertical_count-1)]

    # The last row and column of tiles might not fit
    # (Might go outside the image)
    # so get the tile positiion by subtracting tile size from the
    # edge of the image.
    right_x = padded_im.shape[1] - in_tile_shape[1]
    bottom_y = padded_im.shape[0] - in_tile_shape[0]

    y_coords.append(bottom_y)
    x_coords.append(right_x)

    # because its a rectangle get all combinations of x and y
    tile_coords = [(x, y) for x in x_coords for y in y_coords]
    return tile_coords


def seg_to_rgba(seg, classes_rgba):
    # input class preds are 0-1 and are the output from
    # the CNN before thresholding etc
    # rgb image
    class_preds = np.argmax(seg, 0)
    assert class_preds.shape == (seg.shape[1:])
    rgba_output = np.zeros((list(class_preds.shape[:2]) + [4]))
    # take the class predictions.
    # get the maximum.
    # assign this to be RGB
    # channel for each class
    for i, c in enumerate(classes_rgba):
        class_map = class_preds == i
        rgba_output[class_map] = c
    return rgba_output


def save_then_move(out_path, seg_alpha):
    """ need to save in a temp folder first and
        then move to the segmentation folder after saving
        this is because scripts are monitoring the segmentation folder
        and the file saving takes time..
        We don't want the scripts that monitor the segmentation
        folder to try loading the file half way through saving
        as this causes errors. Thus we save and then rename.
    """
    fname = os.path.basename(out_path)
    temp_path = os.path.join('/tmp', fname)
    imsave(temp_path, seg_alpha)
    shutil.move(temp_path, out_path)

def load_image(photo_path):
    photo = imread(photo_path)
    # sometimes photo is a list where first element is the photo
    if len(photo.shape) == 1:
        photo = photo[0]
    # if 4 channels then convert to rgb
    # (presuming 4th channel is alpha channel)
    if len(photo.shape) > 2 and photo.shape[2] == 4:
        photo = color.rgba2rgb(photo)

    # if image is black and white then change it to rgb
    # TODO: train directly on B/W instead of doing this conversion.
    if len(photo.shape) == 2:
        photo = color.gray2rgb(photo)
    return photo
