"""
Copyright (C) 2019, 2020 Abraham George Smith

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

# pylint: disable=C0111, R0913, R0903, R0914, W0511
import random
import math
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from PIL import Image
from skimage.io import imread
from skimage import img_as_float32, img_as_ubyte
from skimage.exposure import rescale_intensity

from im_utils import load_train_image_and_annot, annot_to_target_and_mask
from file_utils import ls
import im_utils
import elastic

def elastic_transform(photo, annot):
    def_map = elastic.get_elastic_map(photo.shape,
                                      scale=random.random(),
                                      intensity=0.4 + (0.6 * random.random()))
    photo = elastic.transform_image(photo, def_map)
    annot = elastic.transform_image(annot, def_map, channels=3)
    assert np.sum(annot) == np.sum(np.round(annot)), 'should only consist of ints'
    annot = annot.astype(np.ubyte)
    return photo, annot

def guassian_noise_transform(photo, annot):
    sigma = np.abs(np.random.normal(0, scale=0.09))
    photo = im_utils.add_gaussian_noise(photo, sigma)
    return photo, annot

def salt_pepper_transform(photo, annot):
    salt_intensity = np.abs(np.random.normal(0.0, 0.008))
    photo = im_utils.add_salt_pepper(photo, salt_intensity)
    return photo, annot


class UNetTransformer():
    """ Data Augmentation """
    def __init__(self):
        self.color_jit = ColorJitter(brightness=0.3, contrast=0.3,
                                     saturation=0.2, hue=0.001)

    def transform(self, photo, annot):

        transforms = random.sample([elastic_transform,
                                    guassian_noise_transform,
                                    salt_pepper_transform,
                                    self.color_jit_transform], 4)

        for transform in transforms:
            if random.random() < 0.8:
                photo, annot = transform(photo, annot)

        if random.random() < 0.5:
            photo = np.fliplr(photo)
            annot = np.fliplr(annot)

        return photo, annot

    def color_jit_transform(self, photo, annot):
        # TODO check skimage docs for something cleaner to convert
        # from float to int
        photo = rescale_intensity(photo, out_range=(0, 255))
        photo = Image.fromarray((photo).astype(np.int8), mode='RGB')
        photo = self.color_jit(photo)  # returns PIL image
        photo = img_as_float32(np.array(photo))  # return back to numpy
        return photo, annot


class RPDataset(Dataset):
    def __init__(self, annot_dir, dataset_dir, in_w, out_w, classes,
                 mode, val_tile_refs=None):
        """
        in_w and out_w are the tile size in pixels

        target_classes is a list of the possible output classes
            the position in the list is the index (target) to be predicted by
            the network in the output.
            The value of the elmenent is the rgba (int, int, int) used to draw this 
            class in the annotation.
        """
        self.mode = mode
        target_classes = [c[1][:3] for c in classes]
        self.in_w = in_w
        self.out_w = out_w
        self.annot_dir = annot_dir
        self.target_classes = target_classes
        self.dataset_dir = dataset_dir
        self.val_tiles_refs = val_tile_refs
        if self.mode == 'train':
            self.augmentor = UNetTransformer()

    def __len__(self):
        if self.mode == 'val':
            return len(self.val_tiles_refs)
        else:
            # use at least 612 but when dataset gets bigger start to expand
            # to prevent validation from taking all the time (relatively)
            return max(612, len(ls(self.annot_dir)) * 2)


    def get_train_item(self, i):
        # todo this is a little too random for validation.
        image, annot, _ = load_train_image_and_annot(self.dataset_dir,
                                                     self.annot_dir)
        tile_pad = (self.in_w - self.out_w) // 2

        # ensures each pixel is sampled with equal chance
        im_pad_w = self.out_w + tile_pad
        padded_w = image.shape[1] + (im_pad_w * 2)
        padded_h = image.shape[0] + (im_pad_w * 2)
        padded_im = im_utils.pad(image, im_pad_w)

        # This speeds up the padding.
        annot = annot[:, :, :3]
        padded_annot = im_utils.pad(annot, im_pad_w)
        right_lim = padded_w - self.in_w
        bottom_lim = padded_h - self.in_w

        # TODO:
        # Images with less annoations will still give the same number of
        # tiles in the training procedure as images with more annotation.
        # Further empirical investigation into effects of
        # instance selection required are required.
        while True:
            x_in = math.floor(random.random() * right_lim)
            y_in = math.floor(random.random() * bottom_lim)
            annot_tile = padded_annot[y_in:y_in+self.in_w,
                                      x_in:x_in+self.in_w]
            if np.sum(annot_tile) > 0:
                break

        im_tile = padded_im[y_in:y_in+self.in_w,
                            x_in:x_in+self.in_w]

        assert annot_tile.shape == (self.in_w, self.in_w, 3), (
            f" shape is {annot_tile.shape}")

        assert im_tile.shape == (self.in_w, self.in_w, 3), (
            f" shape is {im_tile.shape}")

        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        im_tile, annot_tile = self.augmentor.transform(im_tile, annot_tile)
        im_tile = im_utils.normalize_tile(im_tile)

        # Annotion is cropped post augmentation to ensure
        # elastic grid doesn't remove the edges.
        annot_tile = annot_tile[tile_pad:-tile_pad, tile_pad:-tile_pad]
        target, mask = annot_to_target_and_mask(annot_tile, self.target_classes)


        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        target = target.astype(np.int64)
        target = torch.from_numpy(target)

        im_tile = im_tile.astype(np.float32)
        im_tile = np.moveaxis(im_tile, -1, 0)
        im_tile = torch.from_numpy(im_tile)
        return im_tile, target, mask

    def get_val_item(self, tile_ref, i):
        fname, (x, y), _ = tile_ref
        annot_path = os.path.join(self.annot_dir, fname)
        image_path_part = os.path.join(self.dataset_dir, os.path.splitext(fname)[0])
        # it's possible the image has a different extenstion
        # so use glob to get it
        image_path = glob.glob(image_path_part + '.*')[0]
        image = im_utils.load_image(image_path)
        tile_pad = (self.in_w - self.out_w) // 2

        padded_im = im_utils.pad(image, tile_pad)
        im_tile = padded_im[y:y+self.in_w,
                            x:x+self.in_w]
        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        annot = img_as_ubyte(imread(annot_path))
        assert np.sum(annot) > 0
        assert image.shape[2] == 3 # should be RGB
        annot_tile = annot[y:y+self.out_w, x:x+self.out_w]
        target, mask = annot_to_target_and_mask(annot_tile, self.target_classes)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        target = target.astype(np.int64)
        target = torch.from_numpy(target)
        im_tile = im_tile.astype(np.float32)
        im_tile = np.moveaxis(im_tile, -1, 0)
        im_tile = torch.from_numpy(im_tile)
        return im_tile, target, mask

    def __getitem__(self, i):
        if self.mode == 'val':
            return self.get_val_item(self.val_tiles_refs[i], i)
        elif self.mode == 'train':
            return self.get_train_item(i)
