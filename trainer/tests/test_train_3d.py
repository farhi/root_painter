"""
Test training a 3D CNN

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
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
import pytest
import nibabel as nib

from trainer import Trainer
from unet3d import UNet3D
from metrics import get_metrics_from_arrays
from test_utils import create_tmp_sync_dir


@pytest.mark.slow
def test_train_to_predict_zeros_from_random():
    """
    Very simple test to check that we can train the CNN model to predict 0
    We will use the same input (random tensor) every time
    """
    torch.cuda.empty_cache() # we need to make sure we have enough memory for this test.
    num_classes = 3
    cnn = UNet3D(im_channels=1, out_channels=num_classes).cuda()
    test_input = torch.rand((1, 1, 56, 240, 240)).cuda()
    batch_size = 1
    target = torch.zeros((batch_size, 18, 194, 194)).cuda().long()
    outputs = cnn(test_input)
    assert outputs.shape[1] == num_classes
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.99, nesterov=True)
    class_1_pred_sum = torch.sum(outputs[:, 1])
    while class_1_pred_sum > 0:
        optimizer.zero_grad()
        outputs = cnn(test_input)
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        optimizer.step()
        class_1_pred_sum = torch.sum(outputs[:, 1])
    assert class_1_pred_sum == 0

@pytest.mark.slow
def test_train_to_predict_ones_from_random():
    """
    Very simple test to check that we can train the CNN model to predict 1
    We will use the same input (random tensor) every time
    """
    num_classes = 31
    cnn = UNet3D(im_channels=1, out_channels=num_classes).cuda()
    batch_size = 1
    test_input = torch.rand((batch_size, 1, 56, 240, 240)).cuda()
    target = torch.ones((batch_size, 18, 194, 194)).cuda().long()
    outputs = cnn(test_input)
    assert outputs.shape[1] == num_classes
    assert outputs.shape[0] == batch_size
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.99, nesterov=True)
    for _ in range(100):
        optimizer.zero_grad()
        outputs = cnn(test_input)
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        optimizer.step()
        class_preds = torch.argmax(outputs, dim=1) # class predictions
        class1_preds_sum = torch.sum(class_preds)
        diff_from_1s = abs(class1_preds_sum - torch.sum(target))
        unique = torch.unique(class_preds)
        if len(unique) == 1 and diff_from_1s == 0:
            print('predicted all 1s')
            return
    assert False, 'test took too long'


@pytest.mark.slow
def test_train_identity_from_random():
    """
    Very simple test to check that we can train the CNN model to
    map input to output. Changing random numbers.

    It cannot actually do this but we can test that it gets close.
    """
    torch.cuda.empty_cache() # we need to make sure we have enough memory for this test.
    cnn = UNet3D(im_channels=1, out_channels=1).cuda()
    batch_size = 1
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.99, nesterov=True)
    criterion = torch.nn.MSELoss() # see [1]

    for _ in range(100):
        # small input for this test to make it go faster.
        test_input = torch.rand((batch_size, 1, 56, 56, 56)).cuda()
        target = test_input[:, :, 19:-19, 19:-19, 19:-19]

        optimizer.zero_grad()
        outputs = cnn(test_input)
        # Also consider MarginRankingLoss for similar problems, see [0]
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        diff_mean = torch.mean(torch.abs(outputs - target))
        if diff_mean < 0.1:
            return
    assert False, 'test took too long'


def load_nifty(image_path):
    """ load compressed nifty file from disk and
        switch first and last channel
    """
    image = nib.load(image_path)
    image = np.array(image.dataobj)
    image = np.moveaxis(image, -1, 0) # depth moved to beginning
    return image


def load_heart_patch(data_dir, filter_labels=True):
    """ Load a heart from the struct seg data.
        Include binary (0,1) labels for the voxels which contain the heart.
        Return a patch small enough to fit on the GPU (56, 240, 240)
    """
    image = load_nifty(os.path.join(data_dir, 'data.nii.gz'))
    annot = load_nifty(os.path.join(data_dir, 'label.nii.gz'))

    image_patch = image[12:12+56, 150:150+240, 150:150+240]
    annot_patch = annot[12:12+56, 150:150+240, 150:150+240]
    # it's only the middle bit due to valid padding
    annot_patch = annot_patch[19:-19, 23:-23, 23:-23]

    if filter_labels:
        heart_labels = (annot_patch == 3).astype(np.int16) # lets keep it simple at the start
        return image_patch, heart_labels

    return image_patch, annot_patch.astype(np.int16)



@pytest.mark.slow
def test_train_struct_seg_heart_patch():
    """
    Test training CNN model to predict heart in a single struct seg patch.

    This test requires the struct seg dataset has been downloaded to the users
    home folder otherwise it will be skipped.
    """

    torch.cuda.empty_cache() # we need to make sure we have enough memory for this test.
    data_dir = os.path.join(Path.home(), 'datasets', 'Thoracic_OAR', '1')
    if not os.path.isdir(data_dir):
        print('skip test as data not found')
        return

    cnn = UNet3D(im_channels=1, out_channels=2).cuda() # heart / not heart
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.99, nesterov=True)

    # get data and heart labels for a patch with the heart in it
    image_patch, heart_labels = load_heart_patch(data_dir)

    # Add chanel dimension, I guess this could be useful for multi-modality
    image_patch = np.expand_dims(image_patch, axis=0)

    # Add batch dimension
    image_patch = np.expand_dims(image_patch, axis=0)
    heart_labels = np.expand_dims(heart_labels, axis=0)

    # prepare for gpu
    image_patch = torch.from_numpy(image_patch).cuda().float()
    heart_labels = torch.from_numpy(heart_labels).cuda().long()

    # 25 steps should be enough to get at least 0.9 dice
    for i in range(35):
        optimizer.zero_grad()
        outputs = cnn(image_patch)
        loss = F.cross_entropy(outputs, heart_labels)
        loss.backward()
        optimizer.step()
        heart_preds = F.softmax(outputs, 1)[0].detach()
        heart_preds = torch.argmax(heart_preds, axis=0).cpu().numpy()
        dice = get_metrics_from_arrays(heart_preds,
                                       heart_labels.cpu().numpy())['dice']
        print(f'Fitting single heart patch. {i} dice:{dice}, loss: {loss.item()}')
        if dice > 0.9:
            return
    assert False, 'Takes too long to fit heart patch'


def send_training_instruction(sync_dir):
    """ send a 'start_training' instruction """
    instruction_dir = os.path.join(sync_dir, 'instructions')
    content = {
        "dataset_dir": os.path.join(sync_dir, 'dataset'),
        "model_dir": os.path.join(sync_dir, 'models'),
        "log_dir": sync_dir,
        "val_annot_dir": os.path.join(sync_dir, 'annots'),
        "train_annot_dir": os.path.join(sync_dir, 'annots'),
        "message_dir": os.path.join(sync_dir, 'messages'),
        "dimensions": 3,
        "classes": ['heart'] # output channels of the network will be 0:heart, 1:not_heart
    }
    hash_str = '_' + str(hash(json.dumps(content)))
    fpath = os.path.join(instruction_dir, 'start_training' + hash_str)
    with open(fpath, 'w') as json_file:
        json.dump(content, json_file, indent=4)



def get_masked_heart_annot(annot):
    """
    Retrun an annotation where only the fg and bg
    around the heart are defined.
    Ths mitigates the class imbalnce problem and allows for faster training.
    """
    heart = (annot == 3).astype(np.int16) # only heart labels this time.
    not_heart = (annot != 3).astype(np.int16) # only heart labels this time.
    heart_locations = np.argwhere(heart > 0)
    min_z = np.min(heart_locations[:, 0])
    min_y = np.min(heart_locations[:, 1])
    min_x = np.min(heart_locations[:, 2])
    max_z = np.max(heart_locations[:, 0])
    max_y = np.max(heart_locations[:, 1])
    max_x = np.max(heart_locations[:, 2])
    # set no-heart region to be a cube around the heart
    heart_cube_mask = np.zeros(heart.shape)
    heart_cube_mask[min_z-0:max_z+0,
                    min_y-100:max_y+100,
                    min_x-100:max_x+100] = 1
    # anything outside the heart cube is set to 0
    not_heart *= heart_cube_mask.astype(np.int16)
    annot = np.stack((not_heart, heart))
    annot_byte = annot.astype(np.byte) # reduce size from 100mb to 50mb
    return annot_byte


@pytest.mark.slow
def test_train_struct_seg_heart_from_image():
    """
    Test training CNN model to predict heart in a single struct seg image

    This test requires the struct seg dataset has been downloaded to the users
    home folder otherwise it will be skipped.
    """
    torch.cuda.empty_cache() # we need to make sure we have enough memory for this test.
    data_dir = os.path.join(Path.home(), 'datasets', 'Thoracic_OAR', '1')
    if not os.path.isdir(data_dir):
        print('skip test as data not found')
        return
    sync_dir = create_tmp_sync_dir()

    # Create a dataset containing a single image
    image = load_nifty(os.path.join(data_dir, 'data.nii.gz'))
    annot = load_nifty(os.path.join(data_dir, 'label.nii.gz'))

    # Eventually we will have two channels for each struture.
    # it could work like this.
    # channel 0 = not heart
    # channel 1 = heart
    # channel 2 = not lungs
    # channel 3 = lungs
    # channel 4 = not esophagus
    # channel 5 = esophagus

    # This is required to allow the user to specify corrections for each
    # structure each channel is a binary map. 0 or 1. Indicating the specific
    # locations where annotation has been defined. For now we will start with
    # just the heart.

    # So
    # channel 0 = not heart
    # channel 1 = heart

    # The struct seg dataset has a large class imblance, with the heart
    # occupying less than 0.7% of the image.  This poses challenges to training
    # that will not occur during corrective annotation as the protocol
    # specifies that the user will aim to label no more than 5 to 10 times as
    # much background as foreground for each structure.

    annot_byte = get_masked_heart_annot(annot)
    annot_path = os.path.join(sync_dir, 'annots', '01.npy')
    np.save(annot_path, annot_byte)
    image_path = os.path.join(sync_dir, 'dataset', '01.npy')
    np.save(image_path, image)

    send_training_instruction(sync_dir)
    trainer = Trainer(sync_dir)

    def on_epoch_end():
        print('epoch end')
        fnames = os.listdir(sync_dir)
        fnames = [f for f in fnames if f.endswith('cur_val.csv')]
        if fnames:
            fpath = os.path.join(sync_dir, fnames[0])
            lines = open(fpath).readlines()
            for line in lines[1:]:
                print(float(line.split(',')[-1]))
                if float(line.split(',')[-1]) > 0.6:
                    print('pass')
                    trainer.running = False
                    return
            assert len(lines) < 10, 'takes too long to fit patch'

    trainer.main_loop(on_epoch_end)

# pylint: disable=W0105 # string has no effect
# pylint: disable=C0301 # line too long
"""
[0] https://discuss.pytorch.org/t/how-to-calculate-pair-wise-differences-between-two-tensors-in-a-vectorized-way/37451/3
[1] https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
"""
