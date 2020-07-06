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

# pylint: disable=C0111, W0511
# pylint: disable=E0401 # import error

import os
import warnings
import glob
import sys

import numpy as np
from skimage import color
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.transform import resize
from PIL import Image
from PyQt5 import QtGui
import qimage2ndarray
import nibabel as nib


def is_image(fname):
    extensions = {".jpg", ".png", ".jpeg", '.tif', '.tiff', '.npy'}
    return any(fname.lower().endswith(ext) for ext in extensions)

def load_image(image_path):
    if image_path.endswith('.npy'):
        return np.load(image_path, mmap_mode='c')
    if image_path.endswith('.nii.gz'):
        image = nib.load(image_path)
        return np.array(image.dataobj)
    image = imread(image_path)
    # sometimes image is a list where first element is the image
    if len(image.shape) == 1:
        image = image[0]
    # if 4 channels then convert to rgb
    # (presuming 4th channel is alpha channel)
    if len(image.shape) > 2 and image.shape[2] == 4:
        return color.rgba2rgb(image)

    # if image is black and white then change it to rgb
    # TODO: train directly on B/W instead of doing this conversion.
    if len(image.shape) == 2:
        return color.gray2rgb(image)
    return image


def norm_slice(img, min_v, max_v, brightness_percent):
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    bright_v = (brightness_percent / 100)
    img[img < min_v] = min_v
    img[img > max_v] = max_v
    img -= min_v
    img /= (max_v - min_v)
    img *= bright_v
    img[img > 1] = 1.0
    img *= 255
    return img


def annot_slice_to_pixmap(slice_np):
    """ convert slice from the numpy annotation data
        to a PyQt5 pixmap object """
    # for now fg and bg colors are hard coded.
    # later we plan to let the user specify these in the user interface.
    np_rgb = np.zeros((slice_np.shape[1], slice_np.shape[2], 4))
    np_rgb[:, :, 1] = slice_np[0] * 255 # green is bg
    np_rgb[:, :, 0] = slice_np[1] * 255 # red is fg
    np_rgb[:, :, 3] = np.sum(slice_np, axis=0) * 180 # alpha is defined
    q_image = qimage2ndarray.array2qimage(np_rgb)
    return QtGui.QPixmap.fromImage(q_image)


def seg_slice_to_pixmap(slice_np):
    """ convert slice from the numpy segmentation data
        to a PyQt5 pixmap object """
    np_rgb = np.zeros((slice_np.shape[0], slice_np.shape[1], 4))
    np_rgb[slice_np > 0] = [0, 255, 255, 180]
    q_image = qimage2ndarray.array2qimage(np_rgb)
    return QtGui.QPixmap.fromImage(q_image)


def gen_composite(annot_dir, photo_dir, comp_dir, fname, ext='.jpg'):
    """ for review.
    Output the pngs with the annotation overlaid next to it.
    should make it possible to identify errors. """
    out_path = os.path.join(comp_dir, fname.replace('.png', ext))
    if not os.path.isfile(out_path):
        name_no_ext = os.path.splitext(fname)[0]
        # doesn't matter what the extension is
        glob_str = os.path.join(photo_dir, name_no_ext) + '.*'
        bg_fpath = list(glob.iglob(glob_str))[0]
        background = load_image(bg_fpath)
        annot = imread(os.path.join(annot_dir, os.path.splitext(fname)[0] + '.png'))
        if sys.platform == 'darwin':
            # resize uses np.linalg.inv and causes a segmentation fault
            # for very large images on osx
            # See https://github.com/bcdev/jpy/issues/139
            # Maybe switch to ATLAS to help (for BLAS)
            # until fixed, use simpler resize method.
            # take every second pixel
            background = background[::2, ::2]
            annot = annot[::2, ::2]
        else:
            background = resize(background,
                                (background.shape[0]//2,
                                 background.shape[1]//2, 3))
            annot = resize(annot, (annot.shape[0]//2, annot.shape[1]//2, 4))

        background = img_as_ubyte(background)
        comp_right = Image.fromarray(background)

        # https://stackoverflow.com/a/55319979 # need to convert annot for pil
        annot = (annot * 255).astype(np.uint8)
        comp_right.paste(Image.fromarray(annot), (0, 0), Image.fromarray(annot[:, :, 3]))
        comp_right = np.array(comp_right)
        # if width is more than 20% bigger than height then vstack
        if background.shape[1] > background.shape[0] * 1.2:
            comp = np.vstack((background, comp_right))
        else:
            comp = np.hstack((background, comp_right))
        assert comp.dtype == np.uint8
        with warnings.catch_warnings():
            # avoid low constrast warning.
            warnings.simplefilter("ignore")
            imsave(out_path, comp, quality=95)


def store_annot_slice(annot_pixmap, annot_data, slice_idx):
    """
    Update .annot_data at slice_idx
    so the values for fg and bg correspond to annot_pixmap)
    """
    slice_rgb_np = qimage2ndarray.rgb_view(annot_pixmap.toImage())
    fg = slice_rgb_np[:, :, 0] > 0
    bg = slice_rgb_np[:, :, 1] > 0
    annot_data[0, slice_idx] = bg
    annot_data[1, slice_idx] = fg
