# ROB 535 Perception Project Task 1

This is team 2's code base for task 1 of the ROB 535 perception project. We implemented separate point cloud and image based methods as described below.

## Point Cloud Based Method

The code for the point cloud based method is located in `pointcloud_processing`.

### Helper Functions

The following helper functions are used throughout the code
- `import_snapshot_random.m` reads in a random snapshot from the training set
- `import_snapshot_specific.m` reads in a snapshot specified by an index
- `read_bin.m` reads data in from a bin file
- `getcuboid.m` takes in a bounding box from the training data and returns a MATLAB cuboid model
- `xyz2organizedpc.m` converts an array of xyz point cloud data into a MATLAB structured point cloud (structured as an M x N x 3 array where M and N are the number of vertical and horizontal angles sampled in the point cloud)

### Code for Understanding the Dataset

The following scripts are used to build an understanding of the training dataset:
- `estimate_lidar_quantities.m` determines the horizontal and vertical angles of the point cloud samples
- `bbox_stats.m` computes various statistics regarding the dimensions of the ground truth bounding boxes
- `visualize_snapshot.m` loads in a random snapshot and plots the data (including multiple representations of the point cloud data)

### Training VoxNet

The script `train_voxnet.m` trains a VoxNet point cloud classifier on the point cloud subsets contained in the ground truth bounding boxes. The script largely follows the [MATLAB tutorial](https://www.mathworks.com/help/vision/ug/train-classification-network-to-classify-object-in-3-d-point-cloud.html) for training VoxNet.

Note that there is also a script `train_ppillars.m` which follows the structure of [MATLAB's tutorial](https://www.mathworks.com/help/deeplearning/ug/lidar-object-detection-using-pointpillars-deep-learning.html) on training a PointPillars network. PointPillars requires intensity for training so in this case the RGB images are converted to grayscale and projected onto the point clouds to act as a pseudo-intensity. This script is only included for completeness and was not used for any submission because the trained network did not work well. 

### Classification

VoxNet discretizes and classifies point clouds, it does not determine regions of interest within a larger point cloud. Therefore a number of steps must be taken to identify point cloud clusters to feed into VoxNet. The process to identify clusters is as follows:

1. segment out ground
2. cluster
3. reject clusters with < N points
4. fit bounding boxes to clusters
5. reject clusters based on various bounding box dimension statistics

The remaining clusters can then be fed to VoxNet. The true cluster may not have been captured through the above process so the predicted label with the maximum confidence is used only if it is greater than a threshold (e.g., 0.95) otherwise label 1 is guessed (because it is the most common label in the training set).

The scripts related to this are as follows:
- `obj_cluster.m` performs the steps listed above and visualized the result
- `test_voxnet.m` runs this classification on the training set
- `submit_voxnet.m` runs this classification on the test set and generated a file for submission

## Image Based Methods

