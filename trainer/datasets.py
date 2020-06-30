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
# pylint: disable=E1101 # Instance of 'tuple' has no 'shape' member (no-member)
# pylint: disable=R0902 # Too many instance attributes (9/7)

import random
import math
import os
import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from skimage.io import imread
from skimage import img_as_float32, img_as_ubyte
from skimage.exposure import rescale_intensity
import torch
import numpy as np

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

def color_jit_transform(photo, annot):
    color_jit = ColorJitter(brightness=0.3, contrast=0.3,
                            saturation=0.2, hue=0.001)
    # TODO check skimage docs for something cleaner to convert
    # from float to int
    photo = rescale_intensity(photo, out_range=(0, 255))
    photo = Image.fromarray((photo).astype(np.int8), mode='RGB')
    photo = color_jit(photo)  # returns PIL image
    photo = img_as_float32(np.array(photo))  # return back to numpy
    return photo, annot

def augment(photo, annot):
    transforms = random.sample([elastic_transform,
                                guassian_noise_transform,
                                salt_pepper_transform,
                                color_jit_transform], 4)
    for transform in transforms:
        if random.random() < 0.8:
            photo, annot = transform(photo, annot)
    if random.random() < 0.5:
        photo = np.fliplr(photo)
        annot = np.fliplr(annot)

    return photo, annot



class RPDataset(Dataset):
    def __init__(self, annot_dir, dataset_dir, in_w, out_w,
                 in_d, out_d, classes, mode, tile_refs=None):
        """
        in_w and out_w are the tile size in pixels

        target_classes is a list of the possible output classes
            the position in the list is the index (target) to be predicted by
            the network in the output.
            The value of the elmenent is the rgba (int, int, int) used to draw this
            class in the annotation.

            When the data is 3D the raw channels (for each class)
            are saved and the RGB values are not necessary.
        """
        self.mode = mode
        self.in_w = in_w
        self.out_w = out_w
        self.in_d = in_d
        self.out_d = out_d
        self.annot_dir = annot_dir
        self.target_classes = [c[1][:3] for c in classes]
        self.dataset_dir = dataset_dir
        self.tile_refs = tile_refs

    def __len__(self):
        if self.mode == 'val':
            return len(self.tile_refs)
        if self.tile_refs is not None:
            return len(self.tile_refs)
        # use at least 612 for 2d or 64 for 3D but when dataset gets
        # bigger start to expand to prevent validation
        # from taking all the time (relatively)
        if self.out_d and self.out_d > 1:
            min_patches = 64
        else:
            min_patches = 612
        return max(min_patches, len(ls(self.annot_dir)) * 2)


    def __getitem__(self, i):
        if self.mode == 'val':
            return self.get_val_item(self.tile_refs[i])
        if self.tile_refs is not None:
            return self.get_train_item(self.tile_refs)
        return self.get_train_item()

    def get_train_item(self, tile_ref=None):
        if self.out_d and self.out_d > 1:
            return self.get_train_item_3d(tile_ref)
        return self.get_train_item_2d()

    def get_random_tile_3d(self, annot, image, pad_w, pad_d):
        # this will find something eventually as we know
        # all annotation contain labels somewhere

        # WARNING this will ever sample the very edge of the iamge
        #(for prediction), as we have fully disabled padding during training.
        padded_d = annot[0].shape[0]
        padded_h = annot[0].shape[1]
        padded_w = annot[0].shape[2]

        right_lim = padded_w - self.in_w
        bottom_lim = padded_h - self.in_w
        depth_lim = padded_d - self.in_d

        while True:
            x_in = math.floor(random.random() * right_lim)
            y_in = math.floor(random.random() * bottom_lim)
            z_in = math.floor(random.random() * depth_lim)
            annot_tile = annot[:,
                               z_in:z_in+self.in_d,
                               y_in:y_in+self.in_w,
                               x_in:x_in+self.in_w]
            annot_tile_center = annot_tile[:, pad_d:-pad_d,
                                           pad_w:-pad_w, pad_w:-pad_w]
            # we only want annotations with defiend regions in the output area
            if np.sum(annot_tile_center) > 0:
                im_tile = image[z_in:z_in+self.in_d,
                                y_in:y_in+self.in_w,
                                x_in:x_in+self.in_w]
                return annot_tile, im_tile


    def get_train_item_3d(self, tile_ref):
        # minimal padding and no augmentation for now.
        # We will bring these back in later.
        # When tile_ref is specified we use these coordinates to get
        # the input tile. Otherwise we will sample randomly

        if tile_ref:
            annot_tile, im_tile, mask = self.get_tile_from_ref_3d(tile_ref)
            # For now just return the tile. We plan to add augmentation here.
            return im_tile, annot_tile, mask

        image, annot, _, _ = load_train_image_and_annot(self.dataset_dir, self.annot_dir)
        pad_width = (self.in_w - self.out_w) // 2
        pad_depth = (self.in_d - self.out_d) // 2
        
        # pad to allow training close to the boundary of the image 
        annot0 = im_utils.pad_3d(annot[0], pad_width, pad_depth)
        annot1 = im_utils.pad_3d(annot[1], pad_width, pad_depth)
        annot = np.stack((annot0, annot1))
        image = im_utils.pad_3d(image, pad_width, pad_depth)
        annot_tile, im_tile = self.get_random_tile_3d(annot, image, pad_width, pad_depth)
        # 3d augmentation isn't implemented properly yet but
        # the annotion should be cropped post augmentation to ensure
        # elastic grid doesn't remove the edges.
        annot_tile = annot_tile[:,
                                pad_depth:-pad_depth,
                                pad_width:-pad_width,
                                pad_width:-pad_width]
        assert np.sum(annot_tile) > 0, 'annot tile should contain annotation'
        assert annot_tile.shape[1:] == (self.out_d, self.out_w, self.out_w), (
            f" shape is {annot_tile.shape}")

        assert im_tile.shape == (self.in_d, self.in_w, self.in_w), (
            f" shape is {im_tile.shape}")

        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        mask = annot_tile[0] + annot_tile[1]
        mask[mask > 1] = 1
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        im_tile = im_tile.astype(np.float32)
        im_tile = torch.from_numpy(np.expand_dims(im_tile, axis=0)) # add channels
        annot_tile = torch.from_numpy(annot_tile).long()
        return im_tile, annot_tile, mask


    def get_train_item_2d(self):
        image, annot, _, _ = load_train_image_and_annot(self.dataset_dir, self.annot_dir)
        # ensures each pixel is sampled with equal chance
        tile_pad = (self.in_w - self.out_w) // 2
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
        im_tile, annot_tile = augment(im_tile, annot_tile)
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

    def get_val_item(self, tile_ref):
        _, coord, _, _ = tile_ref
        if len(coord) == 2:
            return self.get_tile_from_ref_2d(tile_ref)
        return self.get_tile_from_ref_3d(tile_ref)


    def get_tile_from_ref_3d(self, tile_ref):
        fname, (tile_x, tile_y, tile_z), _, _ = tile_ref
        image_path = os.path.join(self.dataset_dir, fname)
        image, _ = im_utils.load_image(image_path)
        pad_width = (self.in_w - self.out_w) // 2
        pad_depth = (self.in_d - self.out_d) // 2
        # padding just incase we are at the edge
        padded_im = im_utils.pad_3d(image, pad_width, pad_depth)
        im_tile = padded_im[tile_z:tile_z+self.in_d,
                            tile_y:tile_y+self.in_w,
                            tile_x:tile_x+self.in_w]
        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        annot_path = os.path.join(self.annot_dir, fname)
        annot = np.load(annot_path, mmap_mode='c')
        annot_tile = annot[:,
                           tile_z:tile_z+self.out_d,
                           tile_y:tile_y+self.out_w,
                           tile_x:tile_x+self.out_w]
        mask = annot_tile[0] + annot_tile[1]
        mask[mask > 1] = 1
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        im_tile = torch.from_numpy(np.expand_dims(im_tile, axis=0))
        annot_tile = torch.from_numpy(annot_tile).long()
        return im_tile, annot_tile, mask


    def get_tile_from_ref_2d(self, tile_ref):
        fname, (tile_x, tile_y), _, _ = tile_ref
        annot_path = os.path.join(self.annot_dir, fname)
        image_path_part = os.path.join(self.dataset_dir, os.path.splitext(fname)[0])
        # it's possible the image has a different extenstion
        # so use glob to get it
        image_path = glob.glob(image_path_part + '.*')[0]
        image = im_utils.load_image(image_path)
        tile_pad = (self.in_w - self.out_w) // 2
        padded_im = im_utils.pad(image, tile_pad)
        im_tile = padded_im[tile_y:tile_y+self.in_w, tile_x:tile_x+self.in_w]
        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        annot = img_as_ubyte(imread(annot_path))
        assert np.sum(annot) > 0
        assert image.shape[2] == 3 # should be RGB
        annot_tile = annot[tile_y:tile_y+self.out_w, tile_x:tile_x+self.out_w]
        target, mask = annot_to_target_and_mask(annot_tile, self.target_classes)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        target = target.astype(np.int64)
        target = torch.from_numpy(target)
        im_tile = im_tile.astype(np.float32)
        im_tile = np.moveaxis(im_tile, -1, 0)
        im_tile = torch.from_numpy(im_tile)
        return im_tile, target, mask
