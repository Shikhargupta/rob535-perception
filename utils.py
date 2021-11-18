import os
import cv2
import json
import torch
import pathlib
import numpy as np
from glob import glob
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from typing import List, Dict


min_size = 600
max_size = 1000
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

def collate_double(batch):
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    """
    x = [sample['x'] for sample in batch]
    y = [sample['y'] for sample in batch]
    x_name = [sample['x_name'] for sample in batch]
    y_name = [sample['y_name'] for sample in batch]
    return x, y, x_name, y_name

def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()

def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.
    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.
    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.
    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[0] = x_scale * bbox[0]
    bbox[1] = y_scale * bbox[1]
    bbox[2] = x_scale * bbox[2]
    bbox[3] = y_scale * bbox[3]
    return bbox

def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.
    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.
    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.
    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
    Returns:
        ~numpy.ndarray: A preprocessed image.
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    normalize = pytorch_normalze
    return normalize(img)

class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        return img, bbox

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
        self.transform = Transform(min_size, max_size)

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
        
        y = np.array([label])

        x_trans = np.transpose(x, (2,0,1))
        x, bboxes = self.transform((x_trans, bboxes))
        x = np.transpose(x,(1,2,0))
        # Create target
        target = {'boxes': bboxes,
                  'labels': y}

        # Convert to tensor
        x = torch.from_numpy(x).type(torch.float32)
        bboxes = torch.from_numpy(bboxes).to(torch.float32)
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




        

        


