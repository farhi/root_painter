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
import nibabel as nib
from file_utils import ls


def is_image(fname):
    """ extensions that have been tested with so far """
    extensions = {".jpg", ".png", ".jpeg", '.tif', '.tiff'}
    fname_ext = os.path.splitext(fname)[1].lower()
    return fname_ext in extensions or fname.endswith('.nii.gz') or fname.endswith('.npy')

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
            fnames = [a for a in fnames if is_image(a)]
            fname = random.sample(fnames, 1)[0]
            annot_path = os.path.join(train_annot_dir, fname)
            dims = 2
            if fname.endswith('.npy'):
                image, _ = load_image(os.path.join(dataset_dir, fname))
                image = np.load(os.path.join(dataset_dir, fname), mmap_mode='c')
                annot = np.load(annot_path, mmap_mode='c')
                dims = 3
            else:
                image_path_part = os.path.join(dataset_dir,
                                               os.path.splitext(fname)[0])
                # it's possible the image has a different extenstion
                # so use glob to get it
                image_path = glob.glob(image_path_part + '.*')[0]
                image = load_image(image_path)
                annot = img_as_ubyte(imread(annot_path))
                assert image.shape[2] == 3 # should be RGB
            assert np.sum(annot) > 0 
            # also return fname for debugging purposes.
            return image, annot, fname, dims
        except Exception as e:
            raise e
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


def pad_3d(image, width, depth, mode='reflect', constant_values=0):
    pad_shape = [(depth, depth), (width, width), (width, width)]
    if len(image.shape) == 4:
        # assume channels first for 4 dimensional data.
        # don't pad channels
        pad_shape = [(0, 0)] + pad_shape
    if mode == 'reflect':
        return skim_util.pad(image, pad_shape, mode)
    return skim_util.pad(image, pad_shape, mode=mode,
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


def get_val_tile_refs(annot_dir, prev_tile_refs, in_w, out_w, dims=2, in_d=None, out_d=None):
    """
    Get tile info which covers all annotated regions of the annotation dataset.
    The list must be structured such that an index can be used to refer to each example
    so that it can be used with a dataloader.

    returns tile_refs (list)
        Each element of tile_refs is a list that includes:
            * image file name (string) - for loading the image from disk during validation 
            * coord (x int, y int) - for addressing the location within the padded image
            * annot_mtime (int)
                The image annotation may get updated by the user at any time.
                We can use the mtime to check for this.
                If the annotation has changed then we need to retrieve tile 
                coords for this image again. The reason for this is that we
                only want tile coords with annotations in. The user may have added or removed
                annotation in part of an image. This could mean a different set of coords (or
                not) should be returned for this image.

    Parameter prev_tile_refs is used for comparing both file names and mtime.

    The annot_dir folder should be checked for any new files (not in prev_tile_refs) or files with
    an mtime different from prev_tile_refs. For these file, the image should be loaded and new tile_refs
    should be retrieved. For all other images the tile_refs from prev_tile_refs can be used.
    """
    tile_refs = []
    cur_annot_fnames = ls(annot_dir)
    prev_annot_fnames = [r[0] for r in prev_tile_refs]
    all_annot_fnames = set(cur_annot_fnames + prev_annot_fnames)

    for fname in all_annot_fnames: 
        # get existing coord refs for this image
        prev_refs = [r for r in prev_tile_refs if r[0] == fname]
        prev_mtimes = [r[2] for r in prev_tile_refs if r[0] == fname]
        need_new_refs = False
        # if no refs for this image then check again
        if not prev_refs:
            need_new_refs = True
        else:
            # otherwise check the modified time of the refs against the file.
            prev_mtime = prev_mtimes[0]
            cur_mtime = os.path.getmtime(os.path.join(annot_dir, fname))

            # if file has been updated then get new refs
            if cur_mtime > prev_mtime:
                need_new_refs = True
        if need_new_refs:
            if in_d:
                new_file_refs = get_val_tile_refs_for_annot_3d(annot_dir, fname,   
                                                               in_w, out_w, in_d, out_d)
            else:
                new_file_refs = get_val_tile_refs_for_annot(annot_dir, fname, in_w, out_w)
            tile_refs += new_file_refs
        else:
            tile_refs += prev_refs
    return tile_refs


def get_val_tile_refs_for_annot(annot_dir, annot_fname, in_w, out_w):
    width_diff = in_w - out_w
    pad_width = width_diff // 2
    annot_path = os.path.join(annot_dir, annot_fname)
    annot = img_as_ubyte(imread(annot_path))
    new_file_refs = []
    annot_path = os.path.join(annot_dir, annot_fname)
    padded_im_shape = (annot.shape[0] + (pad_width * 2), 
                       annot.shape[1] + (pad_width * 2))
    out_tile_shape = (out_w, out_w)
    coords = get_coords(padded_im_shape, annot.shape,
                        in_tile_shape=(in_w, in_w, 3),
                        out_tile_shape=out_tile_shape)
    mtime = os.path.getmtime(annot_path)
    for (x, y) in coords:
        annot_tile = annot[y:y+out_w, x:x+out_w]
        # we only want to validate on annotation tiles
        # which have annotation information.
        if np.sum(annot_tile):
            # fname, [x, y], mtime, metrics i.e [tp, tn, fp, fn]
            new_file_refs.append([annot_fname, [x, y], mtime, None])
    return new_file_refs

def get_val_tile_refs_for_annot_3d(annot_dir, annot_fname, in_w, out_w, in_d, out_d):
    """
    Each element of tile_refs is a list that includes:
        * image file name (string) - for loading the image from disk during validation 
        * coord (x int, y int) - for addressing the location within the padded image
        * annot_mtime (int)
        * cached performance for this tile with previous (current best) model.
          Initialized to None but otherwise [tp, fp, tn, fn]
    """
    width_diff = in_w - out_w
    pad_width = width_diff // 2
    annot_path = os.path.join(annot_dir, annot_fname)
    annot = np.load(annot_path, mmap_mode='c')
    new_file_refs = []
    annot_shape = annot.shape[1:]
    out_tile_shape = (out_w, out_w)
    coords = get_coords_3d(annot_shape, annot_shape,
                            in_tile_shape=(in_d, in_w, in_w),
                            out_tile_shape=(out_d, out_w, out_w))

    mtime = os.path.getmtime(annot_path)
    for (x, y, z) in coords:
        annot_tile = annot[:, z:z+out_d, y:y+out_w, x:x+out_w]
        # we only want to validate on annotation tiles
        # which have annotation information.
        if np.sum(annot_tile):
            # fname, [x, y, z], mtime, prev model metrics i.e [tp, tn, fp, fn] or None
            new_file_refs.append([annot_fname, [x, y, z], mtime, None])
    return new_file_refs


def get_coords(padded_im_shape, im_shape, in_tile_shape, out_tile_shape):

    horizontal_count = ceil(im_shape[1] / out_tile_shape[1])
    vertical_count = ceil(im_shape[0] / out_tile_shape[0])

    # first split the image based on the tiles that fit
    x_coords = [h*out_tile_shape[1] for h in range(horizontal_count-1)]
    y_coords = [v*out_tile_shape[0] for v in range(vertical_count-1)]

    # The last row and column of tiles might not fit
    # (Might go outside the image)
    # so get the tile positiion by subtracting tile size from the
    # edge of the image.
    right_x = padded_im_shape[1] - in_tile_shape[1]
    bottom_y = padded_im_shape[0] - in_tile_shape[0]

    y_coords.append(bottom_y)
    x_coords.append(right_x)

    # because its a rectangle get all combinations of x and y
    tile_coords = [(x, y) for x in x_coords for y in y_coords]
    return tile_coords


def get_coords_3d(padded_im_shape, im_shape, in_tile_shape, out_tile_shape):
    assert len(im_shape) == 3, str(im_shape) # d, h, w
    depth_count = ceil(im_shape[0] / out_tile_shape[0])
    vertical_count = ceil(im_shape[1] / out_tile_shape[1])
    horizontal_count = ceil(im_shape[2] / out_tile_shape[2])

    # first split the image based on the tiles that fit
    z_coords = [d*out_tile_shape[0] for d in range(depth_count-1)] # z is depth
    y_coords = [v*out_tile_shape[1] for v in range(vertical_count-1)]
    x_coords = [h*out_tile_shape[2] for h in range(horizontal_count-1)]

    # The last row and column of tiles might not fit
    # (Might go outside the image)
    # so get the tile positiion by subtracting tile size from the
    # edge of the image.
    lower_z = padded_im_shape[0] - in_tile_shape[0]
    bottom_y = padded_im_shape[1] - in_tile_shape[1]
    right_x = padded_im_shape[2] - in_tile_shape[2]

    z_coords.append(lower_z)
    y_coords.append(bottom_y)
    x_coords.append(right_x)

    # because its a cuboid get all combinations of x, y and z
    tile_coords = [(x, y, z) for x in x_coords for y in y_coords for z in z_coords]
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


def save_then_move(out_path, seg, dims):
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
    if dims == 2:
        imsave(temp_path, seg)
    elif out_path.endswith('.nii.gz'):
        img = nib.Nifti1Image(seg, np.eye(4))
        img.to_filename(temp_path)
    elif out_path.endswith('.npy'):
        np.save(temp_path, seg)
    else:
        raise Exception(f'Unhandled combination of dims: {dims} and {out_path}')
    shutil.move(temp_path, out_path)


def load_image(image_path):
    dims = None
    if image_path.endswith('.npy'):
        image = np.load(image_path, mmap_mode='c')
        dims = 3
    elif image_path.endswith('.nii.gz'):
        # We don't currently use them during but it's useful to be
        # able to load nifty files directory to give the user
        # more convenient segmentation options.
        image = nib.load(image_path)
        image = np.array(image.dataobj)
        image = np.moveaxis(image, -1, 0) # depth moved to beginning
        dims = 3
    else:
        dims = 2
        image = imread(image_path)
    # sometimes image is a list where first element is the image
    if len(image.shape) == 1:
        image = image[0]
    # if 4 channels then convert to rgb
    # (presuming 4th channel is alpha channel)
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = color.rgba2rgb(image)
    # if image is black and white then change it to rgb
    # TODO: train directly on B/W instead of doing this conversion.
    if len(image.shape) == 2:
        image = color.gray2rgb(image)
    return image, dims
