#! /usr/bin/python3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

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


classes = (
    'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
    'Military', 'Commercial', 'Trains'
)

files = glob('trainval/*/*_image.jpg')
# for idx in range(len(files)):
idx = np.random.randint(0, len(files))
snapshot = files[idx]
print(snapshot)

img = plt.imread(snapshot)

xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
xyz = xyz.reshape([3, -1])
proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
proj.resize([3, 4])

try:
    bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
except FileNotFoundError:
    print('[*] bbox not found.')
    bbox = np.array([], dtype=np.float32)

bbox = bbox.reshape([-1, 11])

uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
uv = uv / uv[2, :]

dist = np.linalg.norm(xyz, axis=0)
fig1 = plt.figure(1, figsize=(16, 9))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.imshow(img)
ax1.scatter(uv[0, :], uv[1, :], c=dist, marker='.', s=1)
ax1.axis('scaled')
fig1.tight_layout()

fig2 = plt.figure(2, figsize=(8, 8))
ax2 = Axes3D(fig2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

step = 5
ax2.scatter(
    xyz[0, ::step], xyz[1, ::step], xyz[2, ::step],
    c=dist[::step], marker='.', s=1
)

colors = ['C{:d}'.format(i) for i in range(10)]
for k, b in enumerate(bbox):
    R = rot(b[0:3])
    t = b[3:6]

    sz = b[6:9]
    vert_3D, edges = get_bbox(-sz / 2, sz / 2)
    vert_3D = R @ vert_3D + t[:, np.newaxis]

    vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
    vert_2D = vert_2D / vert_2D[2, :]

    clr = colors[np.mod(k, len(colors))]

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
    ymax = min(ymax,img.shape[0])
    xmin = max(xmin,0)
    xmax = min(xmax,img.shape[1])

    ax1.plot([xmin, xmax],[ymin, ymin],color='red')

    ax1.plot([xmax, xmax],[ymin, ymax],color='red')

    ax1.plot([xmin, xmax],[ymax, ymax],color='red')

    ax1.plot([xmin, xmin],[ymin, ymax],color='red')
plt.show()

