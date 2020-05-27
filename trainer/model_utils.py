"""
Utilities for working with the U-Net models

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

# pylint: disable=C0111, R0913, R0914, W0511
import os
import time
import glob
import math
import numpy as np
import torch
from torch.nn.functional import softmax
from skimage.io import imread
from skimage import img_as_float32
import im_utils
from unet import UNetGNRes
from metrics import get_metrics
from file_utils import ls


def get_latest_model_paths(model_dir, k):
    fnames = ls(model_dir)
    fnames = sorted(fnames)[-k:]
    fpaths = [os.path.join(model_dir, f) for f in fnames]
    return fpaths


def load_model(model_path, num_classes):
    model = UNetGNRes(out_channels=num_classes)
    try:
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
    model.cuda()
    return model

def create_first_model_with_random_weights(model_dir, num_classes):
    # used when no model was specified on project creation.
    model_num = 1
    model_name = str(model_num).zfill(6)
    model_name += '_' + str(int(round(time.time()))) + '.pkl'
    model = UNetGNRes(out_channels=num_classes)
    model = torch.nn.DataParallel(model)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    model.cuda()
    return model


def get_prev_model(model_dir, num_classes):
    prev_path = get_latest_model_paths(model_dir, k=1)[0]
    prev_model = load_model(prev_path, num_classes)
    return prev_model, prev_path



def save_if_better(model_dir, cur_model, prev_model_path,
                   cur_loss, prev_loss):

    # convert the nans as they don't work in comparison
    if math.isnan(cur_loss):
        cur_loss = 0
    if math.isnan(prev_loss):
        prev_loss = 0
    print('prev loss', str(round(prev_loss, 5)).ljust(7, '0'),
          'cur loss', str(round(cur_loss, 5)).ljust(7, '0'))
    if cur_loss > prev_loss:
        prev_model_fname = os.path.basename(prev_model_path)
        prev_model_num = int(prev_model_fname.split('_')[0])
        model_num = prev_model_num + 1
        now = int(round(time.time()))
        model_name = str(model_num).zfill(6) + '_' + str(now) + '.pkl'
        model_path = os.path.join(model_dir, model_name)
        print('saving', model_path, time.strftime('%H:%M:%S', time.localtime(now)))
        torch.save(cur_model.state_dict(), model_path)
        return True
    return False


def model_file_segment(model_paths, image, bs, in_w, out_w, classes_rgba):
    """ Average predictions from each model specified in model_paths """
    # then add predictions from the previous models to form an ensemble
    cnn = load_model(model_paths[0], len(classes_rgba))
    cnn.half()
    preds = segment(cnn, image, bs, in_w, out_w)
    return im_utils.seg_to_rgba(preds, classes_rgba)


def segment(cnn, image, bs, in_w, out_w):
    # Return prediction for each pixel in the image
    # The cnn will give a the output as channels where
    # each channel corresponds to a specific class 'probability'
    assert image.shape[0] >= in_w, str(image.shape[0])
    assert image.shape[1] >= in_w, str(image.shape[1])

    width_diff = in_w - out_w
    pad_width = width_diff // 2
    padded_im = im_utils.pad(image, pad_width)
    coords = im_utils.get_coords(padded_im, image,
                                 in_tile_shape=(in_w, in_w, 3),
                                 out_tile_shape=(out_w, out_w))
    coord_idx = 0
    batches = []
    
    # segmentation for the full image
    # assign once we get number of classes from the cnn output shape.
    seg = None

    while coord_idx < len(coords):
        tiles_to_process = []
        coords_to_process = []
        for _ in range(bs):
            if coord_idx < len(coords):
                coord = coords[coord_idx]
                x, y = coord
                tile = padded_im[y:y+in_w,
                                 x:x+in_w]
                assert tile.shape[0] == in_w
                assert tile.shape[1] == in_w
                tile = img_as_float32(tile)
                tile = im_utils.normalize_tile(tile)
                tile = np.moveaxis(tile, -1, 0)
                coord_idx += 1
                tiles_to_process.append(tile)
        
                coords_to_process.append(coord)

        tiles_to_process = np.array(tiles_to_process)
        print('tiles_to_process.shape', tiles_to_process.shape)
        tiles_for_gpu = torch.from_numpy(tiles_to_process)
        tiles_for_gpu.cuda()
        tiles_for_gpu = tiles_for_gpu.half()
        tiles_predictions = cnn(tiles_for_gpu)
        pred_np = tiles_predictions.data.cpu().numpy()
        num_classes = pred_np.shape[1] # how many output classes

        if seg is None:
            seg_shape = [num_classes] + list(image.shape[:2])
            seg = np.zeros(seg_shape)

        out_tiles = pred_np.reshape((len(tiles_for_gpu), num_classes, out_w, out_w))
        
        # add the predictions from the gpu to the output segmentation
        # use their correspond coordinates
        for tile, (x, y) in zip(out_tiles, coords_to_process):
            # tile has channels first so move to channels last.
            seg[:, y:y+tile.shape[1], x:x+tile.shape[2]] = tile
    return seg
