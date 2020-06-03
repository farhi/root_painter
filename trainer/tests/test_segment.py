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
import os
import shutil
import json
import numpy as np
import torch
import time
import pytest
from skimage.io import imread, imsave
from unet import UNetGNRes
from unet3d import UNet3D
from model_utils import segment, create_first_model_with_random_weights
import im_utils
from trainer import Trainer


def test_CNN_segment_classes():
    """
    test CNN returns data in the correct shape for a single tile.
    Using random weights this time so the output is not checked.

    The number of output channels should correspond to the classes
    """
    num_classes = 2
    cnn = UNetGNRes(im_channels=3, out_channels=num_classes)
    test_input = torch.zeros((1, 3, 572, 572))
    output = cnn(test_input)
    assert output.shape[1] == num_classes

    num_classes = 31
    cnn = UNetGNRes(im_channels=3, out_channels=num_classes).cuda()
    test_input = torch.zeros((1, 3, 572, 572)).cuda()
    start = time.time()
    output = cnn(test_input)
    print('2d cnn time 31 class', time.time() - start)

    assert output.shape[1] == num_classes


def test_CNN_segment_classes_3D_3_classes():
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

def test_CNN_segment_classes_3D_9_classes():
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
def test_segment_large_2D_image(monkeypatch):
    # segmentation method reconstructs the output properly
    
    def cnn(tile):
        # return input with 3 channels is equivalent of 3 classes.
        return tile

    def mock_normalize_tile(tile):
        # skip this for now to check that input
        # is reconsructed properly
        return tile

    monkeypatch.setattr(im_utils, "normalize_tile", mock_normalize_tile)

    in_w = 600
    out_w = in_w
    channels = 4
    bs = 2
    test_input = np.random.rand(in_w, in_w, channels)
    seg = segment(cnn, test_input, bs, in_w, out_w)
    assert seg.shape == (channels, in_w, in_w) # 3 class with same size as input
    seg = np.moveaxis(seg, 0, -1) # return back to original image shape
    # its not exactly the same due to converion to pytorch 16bit etc
    assert np.allclose(seg, test_input, atol=0.01)


@pytest.mark.slow
def test_2D_segment_instruction():
    """
    A high level integration test for the 
    full 2D image segmentation prcocess.
    We don't check model accuracy here. 
    """
    # create a 'scrap' folder for the test in /tmp 
    # this scrap folder will be used as the sync_directory for this test.
    sync_dir = os.path.join('/tmp', 'test_sync_dir')
    if os.path.isdir(sync_dir):
        shutil.rmtree(sync_dir) 
    os.makedirs(sync_dir)
    
    # create an instructions folder, models folder,  dataset folder
    # and a segmentation folder inside the sync_directory
    instruction_dir = os.path.join(sync_dir, 'instructions')
    dataset_dir = os.path.join(sync_dir, 'dataset')
    seg_dir = os.path.join(sync_dir, 'seg')
    model_dir = os.path.join(sync_dir, 'models')
    for d in [instruction_dir, dataset_dir, model_dir, seg_dir]:
        os.makedirs(d)
    
    # create a model file (random weights is fine)
    # and save the model to the models folder.
    create_first_model_with_random_weights(model_dir, num_classes=3)

    # create an example input image using numpy and save to the datsets folder
    example_image = np.random.rand(1200, 600, 3)
    im_path = os.path.join(dataset_dir, 'example_image.jpeg')
    imsave(im_path, example_image)

    # create an instruction as json.
    content = {
        "dataset_dir": dataset_dir,
        "seg_dir": seg_dir,
        "file_names": ['example_image.jpeg'],
        "model_dir": model_dir,
        "classes": [('bg', [0,180,0,0], 'w'),
                    ('red', [255,0,0,255], '1'),
                    ('blue', [0, 0, 255, 255], '2')]
    }
    # save the instruction (json file) to the instructions folder. 
    hash_str = '_' + str(hash(json.dumps(content)))
    fpath = os.path.join(instruction_dir, 'segment' + hash_str)
    with open(fpath, 'w') as json_file:
        json.dump(content, json_file, indent=4)
 
    # create a trainer object and set the sync directory
    # to be the scrap folder which has been created.
    trainer = Trainer(sync_dir)

    # tell the trainer to check for instructions
    trainer.check_for_instructions()
    print('checking')

    # assert that the segmentation file has been created
    seg_path = os.path.join(seg_dir, 'example_image.png')
    assert os.path.isfile(seg_path)

    # assert that it it is non-zero (random weights don't do this) 
    seg = imread(seg_path)
    assert np.sum(seg) > 0

    # assert that it has the same shape as the input image
    assert seg.shape == (example_image.shape[0], example_image.shape[1], 4)
    # assert that the segment instruction has been deleted.
    assert not os.path.isfile(fpath) 

    # clean up the the scrap folder that was created in /tmp
    shutil.rmtree(sync_dir) 
