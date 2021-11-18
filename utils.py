import os
import cv2
import json
import torch
import pathlib
from glob import glob
import numpy as np
from typing import List, Dict

classes = np.loadtxt('classes.csv', skiprows=1, dtype=str, delimiter=',')
labels = classes[:, 2].astype(np.uint8)

def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """
    Returns a list of files in a directory/path. Uses pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames

def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def rot(n):
    n = np.asarray(n).flatten()
    assert n.shape == (3,)
    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)


def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e

class PrepareDataset(torch.utils.data.Dataset):
    def __init__(self,
                 inputs: List[pathlib.Path],
                 targets: List[pathlib.Path],
                 use_cache: bool = False,
                 convert_to_format: str = None,
                 mapping: Dict = None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            from multiprocessing import Pool
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.use_cache:
            return self.cached_data[idx]
        else:
            input_ID = self.inputs[idx]
            target_ID = self.targets[idx]
            # Load input and target
            x = cv2.imread(input_ID)
            xmin,xmax,ymin,ymax,label = self.get_bb(target_ID,x)

        bboxes = np.array([xmin,ymin,xmax,ymax])
        bboxes = torch.from_numpy(bboxes).to(torch.float32)
        y = np.array([label])

        # Create target
        target = {'boxes': bboxes,
                  'labels': y}

        # Convert to tensor
        x = torch.from_numpy(x).type(torch.float32)
        target['labels'] = torch.from_numpy(target['labels']).type(torch.uint8)

        return {'x': x, 'y': target, 'x_name': self.inputs[idx], 'y_name': self.targets[idx]}


    def get_bb(self,path,x):
        bbox = np.fromfile(path, dtype=np.float32)
        proj = np.fromfile(path.replace('_bbox.bin', '_proj.bin'), dtype=np.float32)
        proj.resize([3, 4])

        b = bbox.reshape([-1, 11])[0]
        R = rot(b[0:3])
        t = b[3:6]

        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]

        vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
        vert_2D = vert_2D / vert_2D[2, :]

        bbox2Dx = []
        bbox2Dy = []
        for e in edges.T:
            # ax1.plot(vert_2D[0, e], vert_2D[1, e], color=clr)
            bbox2Dx.append(vert_2D[0,e][0])
            bbox2Dx.append(vert_2D[0,e][1])
            bbox2Dy.append(vert_2D[1,e][0])
            bbox2Dy.append(vert_2D[1,e][1])


        xmin,xmax = int(min(bbox2Dx)), int(max(bbox2Dx))
        ymin,ymax = int(min(bbox2Dy)), int(max(bbox2Dy))


        ymin = max(ymin,0)
        ymax = min(ymax,x.shape[0])
        xmin = max(xmin,0)
        xmax = min(xmax,x.shape[1])

        class_id = b[9].astype(np.uint8)
        label = labels[class_id]

        return xmin,xmax,ymin,ymax,label




        

        


