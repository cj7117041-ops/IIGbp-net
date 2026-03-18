import random
from skimage.io import imread

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import numpy as np


class SegDataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        boundary_paths=list,
        transform_input=None,
        transform_target=None,
        transform_boundary=None,
        hflip=False,
        vflip=False,
        affine=False,
    ):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.boundary_paths = boundary_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.transform_boundary = transform_boundary

        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]
        boundary_ID = self.boundary_paths[index]

        x, y, z = imread(input_ID), imread(target_ID), imread(boundary_ID)
        if len(y.shape) == 2:
            y = np.repeat(y[:, :, np.newaxis], 3, axis=2)
        if len(z.shape) == 2:
            z = np.repeat(z[:, :, np.newaxis], 3, axis=2)
        x = self.transform_input(x)
        y = self.transform_target(y)
        z = self.transform_boundary(z)

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                # a = random.uniform(0.0, 1.0)
                x = TF.hflip(x)
                y = TF.hflip(y)
                z = TF.hflip(z)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                # b = random.uniform(0.0, 1.0)
                x = TF.vflip(x)
                y = TF.vflip(y)
                z = TF.vflip(z)

        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-352 / 8, 352 / 8)
            v_trans = random.uniform(-352 / 8, 352 / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            x = TF.affine(x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            y = TF.affine(y, angle, (h_trans, v_trans), scale, shear, fill=0.0)
            z = TF.affine(z, angle, (h_trans, v_trans), scale, shear, fill=0.0)
        return x.float(), y.float(), z.float()

