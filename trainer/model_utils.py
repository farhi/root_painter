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

def load_model(model_path):
    model = UNetGNRes()
    try:
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
    model.cuda()
    return model

def create_first_model_with_random_weights(model_dir):
    # used when no model was specified on project creation.
    model_num = 1
    model_name = str(model_num).zfill(6)
    model_name += '_' + str(int(round(time.time()))) + '.pkl'
    model = UNetGNRes()
    model = torch.nn.DataParallel(model)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    model.cuda()
    return model


def get_prev_model(model_dir):
    prev_path = get_latest_model_paths(model_dir, k=1)[0]
    prev_model = load_model(prev_path)
    return prev_model, prev_path


def multi_class_metrics(get_val_annots, get_seg, classes_rgb) -> list:
    """
    What information does this function need?

    We need to be able to make predictions using the CNN for a full image.
    I.e not just part of the image.
    """
    class_metrics = [{'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}]
    class_metrics *= len(classes_rgb)

    # for each image 
    for fname, annot in get_val_annots():

        # remove parts where annotation is not defined e.g alhpa=0
        a_channel = annot[:, :, 3]
        y_defined = (a_channel > 0).astype(np.int).reshape(-1)

        # load sed, returns a channel for each class
        seg = get_seg(fname)
        
        # for each class
        for i, class_rgb in enumerate(classes_rgb):
            y_true = im_utils.get_class_map(annot, class_rgb)
            y_pred = seg[i]
            
            # only compute metrics on regions where annotation is defined.
            y_true = y_true.reshape(-1)[y_defined > 0]
            y_pred = y_pred.reshape(-1)[y_defined > 0]

            class_metrics[i]['tp'] += np.sum(np.logical_and(y_pred == 1,
                                                            y_true == 1))
            class_metrics[i]['tn'] += np.sum(np.logical_and(y_pred == 0,
                                                            y_true == 0))
            class_metrics[i]['fp'] += np.sum(np.logical_and(y_pred == 1,
                                                            y_true == 0))
            class_metrics[i]['fn'] += np.sum(np.logical_and(y_pred == 0,
                                                            y_true == 1))
    for i, m in enumerate(class_metrics):
        class_metrics[i] = get_metrics(m['tp'], m['fp'], m['tn'], m['fn'])
    return class_metrics


def get_val_metrics(cnn, val_annot_dir, dataset_dir, in_w, out_w, bs):

    start = time.time()
    fnames = ls(val_annot_dir)
    fnames = [a for a in fnames if im_utils.is_photo(a)]
    cnn.half()
    
    def get_seg(fname):
        image_path_part = os.path.join(dataset_dir, os.path.splitext(fname)[0])
        image_path = glob.glob(image_path_part + '.*')[0]
        image = im_utils.load_image(image_path)
        predicted = segment(cnn, image, bs, in_w, out_w)
        return predicted

    def get_val_annots(fname):
        for fname in fnames:
            annot_path = os.path.join(val_annot_dir,
                                      os.path.splitext(fname)[0] + '.png')
            annot = imread(annot_path)
            annot = np.array(annot)
            yield [fname, annot]

    print('Validation duration', time.time() - start)
    return multi_class_metrics(get_val_annots, get_seg, classes_rgb)


def save_if_better(model_dir, cur_model, prev_model_path,
                   cur_f1, prev_f1):
    print('prev f1', str(round(prev_f1, 5)).ljust(7, '0'),
          'cur f1', str(round(cur_f1, 5)).ljust(7, '0'))
    if cur_f1 > prev_f1:
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

def model_file_segment(model_paths, image, bs, in_w, out_w, classes)
    """ Average predictions from each model specified in model_paths """
    pred_sum = None
    pred_count = 0
    # then add predictions from the previous models to form an ensemble
    cnn = load_model(model_paths[0])
    cnn.half()
    preds = segment(cnn, image, bs, in_w, out_w, classes)
    return predicted

def segment(cnn, image, bs, in_w, out_w, classes_rgb):
    """
    Return image for each class.
    And return RGB representation using classes_rgb
    """
    assert image.shape[0] > in_w, str(image.shape[0])
    assert image.shape[1] > in_w, str(image.shape[1])

    tiles, coords = im_utils.get_tiles(image,
                                       in_tile_shape=(in_w, in_w, 3),
                                       out_tile_shape=(out_w, out_w))
    tile_idx = 0
    batches = []
    while tile_idx < len(tiles):
        tiles_to_process = []
        for _ in range(bs):
            if tile_idx < len(tiles):
                tile = tiles[tile_idx]
                tile = img_as_float32(tile)
                tile = im_utils.normalize_tile(tile)
                tile = np.moveaxis(tile, -1, 0)
                tile_idx += 1
                tiles_to_process.append(tile)
        tiles_for_gpu = torch.from_numpy(np.array(tiles_to_process))
        tiles_for_gpu.cuda()
        tiles_for_gpu = tiles_for_gpu.half()
        batches.append(tiles_for_gpu)

    output_tiles = []
    for gpu_tiles in batches:
        outputs = cnn(gpu_tiles)
        _, pred_classes = torch.max(outputs, 1)
        pred_np = pred_classes.data.cpu().numpy()
        out_tiles = pred_np.reshape((len(gpu_tiles), out_w, out_w))
        for out_tile in out_tiles:
            output_tiles.append(out_tile)

    assert len(output_tiles) == len(coords), (
        f'{len(output_tiles)} {len(coords)}')

    reconstructed = im_utils.reconstruct_from_tiles(output_tiles, coords,
                                                    image.shape[:-1])
    # rgb image
    rgb_output = np.array(list(reconstructed.shape) + [4])
    # channel for each class
    class_preds = np.array(([classes_rgb] + list(reconstructed.shape)))
    for i, c in enumerate(classes_rgb):
        class_preds[i] = (reconstructed == i)
        rgb_output[class_preds[i]] = c
    return class_preds, rgb_output
