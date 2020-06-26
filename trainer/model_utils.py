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
from skimage.io import imread
from skimage import img_as_float32
import im_utils
from unet import UNetGNRes
from unet3d import UNet3D
from metrics import get_metrics
from file_utils import ls


def get_latest_model_paths(model_dir, k):
    fnames = ls(model_dir)
    fnames = sorted(fnames)[-k:]
    fpaths = [os.path.join(model_dir, f) for f in fnames]
    return fpaths


def load_model(model_path, num_classes, dimensions):
    if dimensions == 2:
        model = UNetGNRes(out_channels=num_classes * 2)
    else:
        model = UNet3D(im_channels=1, out_channels=num_classes*2)
    try:
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model)
    # pylint: disable=broad-except, bare-except
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
    model.cuda()
    return model

def create_first_model_with_random_weights(model_dir, num_classes, dimensions):
    # used when no model was specified on project creation.
    model_num = 1
    model_name = str(model_num).zfill(6)
    model_name += '_' + str(int(round(time.time()))) + '.pkl'
    # num out channels is twice number of channels
    # as we have a positive and negative output for each structure.
    if dimensions == 2:
        model = UNetGNRes(out_channels=num_classes*2)
    elif dimensions == 3:
        model = UNet3D(im_channels=1, out_channels=num_classes*2)
    else:
        raise Exception(f"Unhandled dimensions {dimensions}")

    model = torch.nn.DataParallel(model)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    model.cuda()
    return model


def get_prev_model(model_dir, num_classes, dims):
    prev_path = get_latest_model_paths(model_dir, k=1)[0]
    prev_model = load_model(prev_path, num_classes, dims)
    return prev_model, prev_path

def get_class_metrics(get_val_annots, get_seg, classes) -> list:
    """
    Segment the validation images and
    return metrics for each of the classes.
    """
    class_metrics = [{'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'class': c} for c in classes]
    classes_rgb = [c[1][:3] for c in classes]

    # for each image
    for fname, annot in get_val_annots():
        assert annot.dtype == np.ubyte, str(annot.dtype)

        # remove parts where annotation is not defined e.g alhpa=0
        a_channel = annot[:, :, 3]
        y_defined = (a_channel > 0).astype(np.int).reshape(-1)

        # load sed, returns a channel for each class
        seg = get_seg(fname)

        # for each class
        for i, class_rgb in enumerate(classes_rgb):
            y_true = im_utils.get_class_map(annot, class_rgb)
            y_pred = seg == i
            assert y_true.shape == y_pred.shape, str(y_true.shape) + str(y_pred.shape)

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
    for i, metric in enumerate(class_metrics):
        class_metrics[i] = get_metrics(metric['tp'], metric['fp'],
                                       metric['tn'], metric['fn'])
    return class_metrics


def get_val_metrics(cnn, val_annot_dir, dataset_dir, in_w, out_w, batch_size, classes):
    # This is no longer used.
    start = time.time()
    fnames = ls(val_annot_dir)
    fnames = [a for a in fnames if im_utils.is_photo(a)]
    cnn.half()

    def get_seg(fname):
        image_path_part = os.path.join(dataset_dir, os.path.splitext(fname)[0])
        image_path = glob.glob(image_path_part + '.*')[0]
        image = im_utils.load_image(image_path)
        predicted = segment(cnn, image, batch_size, in_w, out_w)

        # Need to convert to predicted class.
        predicted = np.argmax(predicted, 0)
        return predicted

    def get_val_annots():
        for fname in fnames:
            annot_path = os.path.join(val_annot_dir,
                                      os.path.splitext(fname)[0] + '.png')
            annot = imread(annot_path)
            annot = np.array(annot)
            yield [fname, annot]

    print('Validation duration', time.time() - start)
    return get_class_metrics(get_val_annots, get_seg, classes)



def save_if_better(model_dir, cur_model, prev_model_path, cur_dice, prev_dice):

    # convert the nans as they don't work in comparison
    if math.isnan(cur_dice):
        cur_dice = 0
    if math.isnan(prev_dice):
        prev_dice = 0
    print('prev dice', str(round(prev_dice, 5)).ljust(7, '0'),
          'cur dice', str(round(cur_dice, 5)).ljust(7, '0'))
    if cur_dice > prev_dice:
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


def ensemble_segment_2d(model_paths, image, batch_size, in_w, out_w, classes_rgba):
    """ Average predictions from each model specified in model_paths """
    pred_sum = None
    # then add predictions from the previous models to form an ensemble
    for model_path in model_paths:
        # load model automatically doubles the number of classes so use half for 2d
        # as 2d will specify foreground and background explicitly,
        # where as this is assumed implicity with 3d
        cnn = load_model(model_path, len(classes_rgba) // 2, dimensions=2)
        cnn.half()
        preds = segment(cnn, image, batch_size, in_w, out_w)
        if pred_sum is not None:
            pred_sum += preds
        else:
            pred_sum = preds
        # get flipped version too (test time augmentation)
        flipped_im = np.fliplr(image)
        flipped_pred = segment(cnn, flipped_im, batch_size, in_w, out_w)
        pred_sum += np.flip(flipped_pred, 2) # return to normal
    return im_utils.seg_to_rgba(pred_sum, classes_rgba)


def ensemble_segment_3d(model_paths, image, batch_size, in_w, out_w, in_d,
                        out_d, num_classes, threshold=0.5, aug=True):
    """ Average predictions from each model specified in model_paths """
    pred_sum = None
    pred_count = 0
    # then add predictions from the previous models to form an ensemble
    for model_path in model_paths:
        cnn = load_model(model_path, num_classes, dimensions=3)
        cnn.half()
        preds = segment_3d(cnn, image, batch_size, in_w, out_w, in_d, out_d)
        if pred_sum is not None:
            pred_sum += preds
            pred_count += 1
        else:
            pred_sum = preds
            pred_count = 1
        if aug:
            # get flipped version too (test time augmentation)
            flipped_im = np.fliplr(image)
            flipped_pred = segment_3d(cnn, flipped_im, batch_size, in_w, out_w, in_d, out_d)
            pred_sum += np.flip(flipped_pred, 2) # return to normal
            pred_count += 1

    if pred_count > 1:
        pred_sum = pred_sum / pred_count
    if threshold is not None:
        pred_sum = (pred_sum > threshold).astype(np.byte)
    return pred_sum # don't need to divide if only one prediction


def segment_3d(cnn, image, batch_size, in_w, out_w, in_d, out_d):

    # image shape = (depth, height, width)
    # in_w is both w and h
    # out_w is both w and h

    # Return prediction for each pixel in the image
    # The cnn will give a the output as channels where
    # each channel corresponds to a specific class 'probability'
    # don't need channel dimension
    # make sure the width, height and depth is at least as big as the tile.
    assert len(image.shape) == 3, str(image.shape)
    assert image.shape[0] >= in_d, f"{image.shape[0]},{in_d}"
    assert image.shape[1] >= in_w, f"{image.shape[1]},{in_w}"
    assert image.shape[2] >= in_w, f"{image.shape[2]},{in_w}"

    width_diff = in_w - out_w
    pad_width = width_diff // 2

    depth_diff = in_d - out_d
    pad_depth = depth_diff // 2

    padded_im = im_utils.pad_3d(image, pad_width, pad_depth)
    coords = im_utils.get_coords_3d(padded_im.shape, image.shape,
                                    in_tile_shape=(in_d, in_w, in_w),
                                    out_tile_shape=(out_d, out_w, out_w))
    coord_idx = 0

    # segmentation for the full image
    # assign once we get number of classes from the cnn output shape.
    seg = None

    while coord_idx < len(coords):
        tiles_to_process = []
        coords_to_process = []
        for _ in range(batch_size):
            if coord_idx < len(coords):
                coord = coords[coord_idx]
                x_coord, y_coord, z_coord = coord
                tile = padded_im[z_coord:z_coord+in_d,
                                 y_coord:y_coord+in_w,
                                 x_coord:x_coord+in_w]
                # need to add channel dimension for GPU processing.
                tile = np.expand_dims(tile, axis=0)
                assert tile.shape[1] == in_d, str(tile.shape)
                assert tile.shape[2] == in_w, str(tile.shape)
                assert tile.shape[3] == in_w, str(tile.shape)
                tile = img_as_float32(tile)
                tile = im_utils.normalize_tile(tile)
                coord_idx += 1
                tiles_to_process.append(tile) # need channel dimension
                coords_to_process.append(coord)

        tiles_to_process = np.array(tiles_to_process)
        tiles_for_gpu = torch.from_numpy(tiles_to_process)
        tiles_for_gpu.cuda()
        tiles_for_gpu = tiles_for_gpu.half()
        tiles_predictions = cnn(tiles_for_gpu)
        pred_np = tiles_predictions.data.cpu().numpy()

        num_classes = pred_np.shape[1] # how many output classes

        if seg is None:
            seg_shape = [num_classes] + list(image.shape)
            seg = np.zeros(seg_shape)

        out_tiles = pred_np.reshape((len(tiles_for_gpu), num_classes, out_d, out_w, out_w))


        # add the predictions from the gpu to the output segmentation
        # use their correspond coordinates
        for tile, (x_coord, y_coord, z_coord) in zip(out_tiles, coords_to_process):
            # tile.shape[0] is the number of classes.
            seg[:,
                z_coord:z_coord+tile.shape[1],
                y_coord:y_coord+tile.shape[2],
                x_coord:x_coord+tile.shape[3]] = tile
    return seg

def segment(cnn, image, batch_size, in_w, out_w):
    # Return prediction for each pixel in the image
    # The cnn will give a the output as channels where
    # each channel corresponds to a specific class 'probability'
    assert image.shape[0] >= in_w, str(image.shape[0])
    assert image.shape[1] >= in_w, str(image.shape[1])

    width_diff = in_w - out_w
    pad_width = width_diff // 2
    padded_im = im_utils.pad(image, pad_width)
    coords = im_utils.get_coords(padded_im.shape, image.shape,
                                 in_tile_shape=(in_w, in_w, 3),
                                 out_tile_shape=(out_w, out_w))
    coord_idx = 0

    # segmentation for the full image
    # assign once we get number of classes from the cnn output shape.
    seg = None

    while coord_idx < len(coords):
        tiles_to_process = []
        coords_to_process = []
        for _ in range(batch_size):
            if coord_idx < len(coords):
                coord = coords[coord_idx]
                x_coord, y_coord = coord
                tile = padded_im[y_coord:y_coord+in_w,
                                 x_coord:x_coord+in_w]
                assert tile.shape[0] == in_w
                assert tile.shape[1] == in_w
                tile = img_as_float32(tile)
                tile = im_utils.normalize_tile(tile)
                tile = np.moveaxis(tile, -1, 0)
                coord_idx += 1
                tiles_to_process.append(tile)
                coords_to_process.append(coord)

        tiles_to_process = np.array(tiles_to_process)
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
        for tile, (x_coord, y_coord) in zip(out_tiles, coords_to_process):
            # tile has channels first so move to channels last.
            seg[:,
                y_coord:y_coord+tile.shape[1],
                x_coord:x_coord+tile.shape[2]] = tile
    return seg
