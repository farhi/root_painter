"""
Test the segment function

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
# pylint: disable=E1136 # usubscriptable shape
import os
import json
import time
import shutil

import numpy as np
import torch
import pytest
import nibabel as nib

from unet3d import UNet3D
from model_utils import segment_3d, create_first_model_with_random_weights
import im_utils
from trainer import Trainer
from test_utils import create_tmp_sync_dir


def test_cnn_segment_classes_3d_3_classes():
    """
    test CNN returns data in the correct shape for a single cube.
    Using random weights this time so the output is not checked.

    The number of output channels should correspond to the classes
    """
    start = time.time()
    num_classes = 3
    cnn = UNet3D(im_channels=1, out_channels=num_classes).cuda()
    test_input = torch.zeros((1, 1, 64, 312, 312)).cuda()
    output = cnn(test_input)
    print('3D unet 3 classes output time = ', time.time() - start, 'output shape', output.shape)
    assert output.shape[1] == num_classes


def test_cnn_segment_classes_3d_9_classes():
    """
    test CNN returns data in the correct shape for a single cube.
    Using random weights this time so the output is not checked.

    The number of output channels should correspond to the classes
    """
    num_classes = 9
    cnn = UNet3D(im_channels=1, out_channels=num_classes).cuda()
    test_input = torch.zeros((1, 1, 64, 312, 312)).cuda()
    start = time.time()
    output = cnn(test_input)
    print('3D unet 9 classes output time = ', time.time() - start, 'output shape', output.shape)
    assert output.shape[1] == num_classes


@pytest.mark.slow
def test_segment_large_3d_image(monkeypatch):
    """ Test that the segmentation method reconstructs the output properly """

    def cnn(tile):
        assert len(tile.shape) == 5
        assert tile.shape[0] == 1 # batch dimension
        assert tile.shape[1] == 1 # channel dimension
        assert tile.shape[2] == 64, str(tile.shape) # depth
        assert tile.shape[3] == 312 # height
        assert tile.shape[4] == 312 # width

        # identify function as we are only testing the reconstruction methods
        # test returning a smaller part of the image as would happen in reality.
        # take the central 18 slices.
        d_crop = (64 - 18) // 2
        w_crop = (312 - 274) // 2
        # taking advatnage of the channel size and class size being equivalent
        central_part = tile[:, :, d_crop:-d_crop, w_crop:-w_crop, w_crop:-w_crop]
        return central_part

    def mock_normalize_tile(tile):
        # skip normalization step.
        return tile

    monkeypatch.setattr(im_utils, "normalize_tile", mock_normalize_tile)
    depth = 96
    test_input = np.random.rand(depth, 512, 512) # depth, height, width
    in_w = 312
    out_w = 274
    in_d = 64
    out_d = 18
    batch_size = 1
    seg = segment_3d(cnn, test_input, batch_size, in_w, out_w, in_d, out_d)
    # not exactly equal because of going via 16bit etc.
    assert np.allclose(seg, test_input, atol=0.01)




def send_segment_instruction(content, sync_dir):
    """
    Add a segmentation instruction json file for the specified sync_dir
    """
    instruction_dir = os.path.join(sync_dir, 'instructions')
    # save the instruction (json file) to the instructions folder.
    hash_str = '_' + str(hash(json.dumps(content)))
    fpath = os.path.join(instruction_dir, 'segment' + hash_str)
    with open(fpath, 'w') as json_file:
        json.dump(content, json_file, indent=4)
    return fpath


@pytest.mark.slow
def test_3d_segment_instruction():
    """
    A high level integration test for the
    full 3D image segmentation prcocess.
    We don't check model accuracy here.
    """
    sync_dir = create_tmp_sync_dir()

    # create a model file (random weights is fine)
    # and save the model to the models folder.
    create_first_model_with_random_weights(os.path.join(sync_dir, 'models'),
                                           num_classes=1, dimensions=3)

    # create an example input image using numpy and save to the datsets folder
    example_image = np.random.rand(512, 512, 96) # similar to what was found with struct seg
    img = nib.Nifti1Image(example_image, np.eye(4))
    img_path = os.path.join(os.path.join(sync_dir, 'dataset'), 'example_image.nii.gz')
    img.to_filename(img_path)

    # create an instruction as json.
    fpath = send_segment_instruction({
        "dataset_dir": os.path.join(sync_dir, 'dataset'),
        "seg_dir": os.path.join(sync_dir, 'seg'),
        "file_names": ['example_image.nii.gz'],
        "model_dir": os.path.join(sync_dir, 'models'),
        "dimensions": 3,
        "classes": ['heart']
    }, sync_dir)

    # create a trainer object and set the sync directory
    # to be the scrap folder which has been created.
    trainer = Trainer(sync_dir)

    # tell the trainer to check for instructions
    trainer.check_for_instructions()

    # assert that the segmentation file has been created
    seg_path = os.path.join(sync_dir, 'seg', 'example_image.nii.gz')
    assert os.path.isfile(seg_path)

    # assert that it it is non-zero (random weights don't do this)
    image = nib.load(seg_path)
    seg = np.array(image.dataobj)

    assert np.sum(seg) > 0

    # assert that it has the same shape as the input image
    # We use depth first now
    assert seg.shape[::-1] == example_image.shape
    # assert that the segment instruction has been deleted.
    assert not os.path.isfile(fpath)

    # clean up the the scrap folder that was created in /tmp
    shutil.rmtree(sync_dir)
