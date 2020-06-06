"""
Test the multiclass dice function.

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
import numpy as np
import torch
from model_utils import get_class_metrics


def test_two_class_dice_perfect_score():
    """
    Test that an annotation and prediction
    that match perfectly will give a dice score of 1.0
    This is the get_class_metrics function
    but using a function with a single class.
    """
    annot = np.zeros((100, 100, 4), dtype=np.ubyte)
    annot[:, :, 3] = 255 # annotation completely defined
    annot[:, :, 0] = 255
    classes_rgb = [[255, 0, 0], [0, 255, 0]]
    val_fname = '001.png'

    def get_seg(fname):
        assert fname == val_fname
        # predict class 1 for everything
        class_1_preds = np.ones((100, 100))
        output = np.zeros((2, 100, 100))
        output[0] = class_1_preds

        # may need to specify in_w, out_w and bs to use cnn
        # convert to predicted class
        return np.argmax(output, 0)

    def get_val_annots():
        return [[val_fname, annot]]

    classes = [[str(i), c] for i, c in enumerate(classes_rgb)]
    all_metrics = get_class_metrics(get_val_annots, get_seg, classes)
    assert len(all_metrics) == 2
    assert np.isclose(all_metrics[0]['dice'], 1.0)
    # this doesn't actually work. no predictions for this class so dice is nan
    # assert np.isclose(all_metrics[1]['dice'], 1.0)


def test_two_class_dice_half_score():
    """
    Test that an annotation and prediction can give 0.5 dice
    """
    annot = np.zeros((100, 100, 4), dtype=np.ubyte)
    annot[:, :, 3] = 255 # all defined
    annot[50:, :, 0] = 255
    classes_rgb = [[255, 0, 0], [0, 0, 0]]
    val_fname = '001.png'

    def get_seg(fname):
        assert fname == val_fname
        # predict class 1 for everything
        class_1_preds = np.ones((100, 100))
        class_1_preds[:, 50:] = 0
        class_2_preds = np.ones((100, 100)) - class_1_preds
        output = np.zeros((2, 100, 100))
        output[0] = class_1_preds
        output[1] = class_2_preds

        # may need to specify in_w, out_w and bs to use cnn
        # convert to predicted class
        return np.argmax(output, 0)

    def get_val_annots():
        return [[val_fname, annot]]

    classes = [[str(i), c] for i, c in enumerate(classes_rgb)]
    all_metrics = get_class_metrics(get_val_annots, get_seg, classes)
    assert len(all_metrics) == 2
    assert np.isclose(all_metrics[0]['dice'], 0.5)
    assert np.isclose(all_metrics[1]['dice'], 0.5)


def test_two_class_dice_half_score_with_undefined():
    """
    Test that an annotation and prediction can give 0.5 dice
    This is a test that the extra undefined region does not alter dice
    """

    width = 200
    image = np.zeros((100, width))
    annot = np.zeros((100, width, 4), dtype=np.ubyte)

    # right half of annotation is undefined e.g 0 alpha
    annot[0:, :100, 3] = 255 # first 100 pixels from left are defined.
    annot[50:, :100, 0] = 255
    classes_rgb = [[255, 0, 0], [0, 0, 0]]
    val_fname = '001.png'

    def get_seg(fname):
        assert fname == val_fname
        # predict class 1 for everything
        class_1_preds = torch.ones(image.shape)
        class_1_preds[:, 50:] = 0
        class_2_preds = torch.ones(image.shape) - class_1_preds
        output = np.zeros((2, 100, width))
        output[0] = class_1_preds
        output[1] = class_2_preds
        # may need to specify in_w, out_w and bs to use cnn
        # convert to predicted class
        return np.argmax(output, 0)

    def get_val_annots():
        return [[val_fname, annot]]


    classes = [[str(i), c] for i, c in enumerate(classes_rgb)]
    all_metrics = get_class_metrics(get_val_annots, get_seg, classes)
    assert len(all_metrics) == 2
    assert np.isclose(all_metrics[0]['dice'], 0.5)
    assert np.isclose(all_metrics[1]['dice'], 0.5)
    assert all_metrics[0]['tn'] < 100*100 # undefined should not count towards true negative


def test_four_class_dice_half_score():
    """
    Test that an annotation and prediction
    that match perfectly will give a dice score of 1.0
    This is the get_class_metrics function
    but using a function with a single class.

    Example data to get 0.5 dice for 4 classes

    True:
    +---+---+----+----+
    | 1 | 2 |  3 |  4 |
    | 1 | 2 |  3 |  4 |
    +---+---+----+----+

    Predicted:
    +---+---+----+----+
    | 1 | 2 |  3 |  4 |
    | 2 | 1 |  4 |  3 |
    +---+---+----+----+
    """

    annot = np.zeros((2, 4, 4), dtype=np.ubyte)
    classes_rgb = [[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]]
    classes = [[str(i), c] for i, c in enumerate(classes_rgb)]
    annot[:, :, 3] = 255 # alpha 255, all pixels defined.
    annot[:, 0, 0] = 10 # top row = class 1
    annot[:, 1, 0] = 20 # second row = class 2
    annot[:, 2, 0] = 30 # bottom row = class 3
    annot[:, 3, 0] = 40 # bottom row = class 4
    val_fname = '001.png'

    def get_seg(fname):
        assert fname == val_fname
        # predict class 1 for everything
        output = np.zeros((4, 2, 4))
        # first dimension is channel e.g class
        output[0, 0, 0] = 1
        output[0, 1, 1] = 1

        output[1, 1, 0] = 1
        output[1, 0, 1] = 1

        output[2, 1, 3] = 1
        output[2, 0, 2] = 1

        output[3, 0, 3] = 1
        output[3, 1, 2] = 1
        # may need to specify in_w, out_w and bs to use cnn
        # convert to predicted class
        return np.argmax(output, 0)

    def get_val_annots():
        return [[val_fname, annot]]
    all_metrics = get_class_metrics(get_val_annots, get_seg, classes)

    assert len(all_metrics) == 4
    assert np.isclose(all_metrics[0]['dice'], 0.5)
    assert np.isclose(all_metrics[1]['dice'], 0.5)
    assert np.isclose(all_metrics[2]['dice'], 0.5)
    assert np.isclose(all_metrics[3]['dice'], 0.5)
