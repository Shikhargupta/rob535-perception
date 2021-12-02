close all

%% Define classes

classes = {'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes', ...
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles', ...
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles', ...
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency', ...
    'Military', 'Commercial', 'Trains'};

%% Import Data

% Import random snapshot
[img, xyz, proj, bbox] = import_snapshot_random();

% Set class
c = classes{int64(bbox(10)) + 1};

%% Plot image

% Compute LIDAR point ranges
dist = vecnorm(xyz);

% Plot image
figure(1)
h = imshow(img);
hold on

% Plot LIDAR points projected into image
uv = proj * [xyz; ones(1, size(xyz, 2))];
uv = uv ./ uv(3, :);
scatter(uv(1, :), uv(2, :), 1, dist, '.')

% Plot vehicle bounding box projected into image
bbox_2d = bboxLidarToCamera(getcuboid(bbox, eye(3)), ...
    cameraIntrinsics([proj(1, 1), proj(2, 2)], [proj(1, 3), proj(2, 3)], size(img, 1, 2)), ...
    rigid3d, 'ProjectedCuboid', true);
pcH = vision.roi.ProjectedCuboid;
pcH.Parent = h.Parent;
pcH.Position = bbox_2d;

% Set figure size and title
set(gcf, 'position', [100, 100, 800, 600])
title(c)

%% Plot point cloud

% Rotate axis to LIDAR convention
rotate_axes = [0 0 1; -1 0 0; 0 -1 0];

% Convert organized point cloud
ptCloud = xyz2organizedpc(xyz);

% Compute ranges to points
dist = sqrt(ptCloud.Location(:, :, 1).^2 + ptCloud.Location(:, :, 1).^2 + ptCloud.Location(:, :, 1).^2);

% Plot point cloud
figure(2)
pcshow(ptCloud.Location, dist)
hold on
xlabel('x')
ylabel('y')
zlabel('z')

% Plot vehicle bounding box
plot(getcuboid(bbox, rotate_axes))

% Set figure title
title(c)

% Print number of points corresponding to the vehicle
idx = findPointsInsideCuboid(getcuboid(bbox, rotate_axes), ptCloud);
length(idx)

%% Plot RGB fused point cloud

ptCloudGray = fuseCameraToLidar(rgb2gray(img), ... % convert to grayscale to resemble reflectance/intensity
    ptCloud, ...
    cameraIntrinsics([proj(1, 1), proj(2, 2)], [proj(1, 3), proj(2, 3)], size(img, 1, 2)), ...
    rigid3d(inv(rotate_axes), [0 0 0]));
ptCloudGray.Intensity = double(ptCloudGray.Color(:, 1));
figure(3)
pcshow(ptCloudGray)
xlabel('x')
ylabel('y')
zlabel('z')

%% Plot Road Detection

verticalFoV = [20, -20];
verticalResolution = length(-20 : 0.4 : 20);
horizontalResolution = 360 / 0.2 + 1;

params = lidarParameters(verticalResolution, verticalFoV,...
    horizontalResolution);

% Segment the ground
groundPtsIdx = segmentGroundSMRF(ptCloud, 'ElevationThreshold', 0.2);

% Extract and plot the ground points
groundPtCloud = select(ptCloud, groundPtsIdx);
groundPtCloud = pcorganize(groundPtCloud, params);
groundPtCloud = ptCloud;

x = groundPtCloud.Location(:,:,1);
y = groundPtCloud.Location(:,:,2);
z = groundPtCloud.Location(:,:,3);
range = sqrt(x.^2 + y.^2 + z.^2);
xp = circshift(x, 1, 2);
yp = circshift(y, 1, 2);
zp = circshift(z, 1, 2);
xn = circshift(x, -1, 2);
yn = circshift(y, -1, 2);
zn = circshift(z, -1, 2);
bumpsy = sqrt((xp - xn).^2 + (yp - yn).^2 + (zp - zn).^2)./x;
xp = circshift(x, 1, 1);
yp = circshift(y, 1, 1);
zp = circshift(z, 1, 1);
xn = circshift(x, -1, 1);
yn = circshift(y, -1, 1);
zn = circshift(z, -1, 1);
bumpsx = sqrt((xp - xn).^2 + (yp - yn).^2 + (zp - zn).^2)./x;
bumps = sqrt(bumpsy.^2 + bumpsx.^2);

figure()
imagesc(range(:, 700:1100, :))
hold on
midpt = (700 + 1100) / 2;
n_rows = size(range, 1);
for i = 1:n_rows
    ind = n_rows + 1 - i;
    dr_dy = diff(range(ind, :));
    jumps = find(abs(dr_dy) > 0.1);
    left_jumps = jumps(jumps < midpt);
    right_jumps = jumps(jumps > midpt);
    if ~isempty(left_jumps)
        left_edge = left_jumps(end);
        scatter(left_edge + 1 - 700, ind, 'ro') 
    end
    if ~isempty(right_jumps)
        right_edge = right_jumps(1);
        scatter(right_edge + 1 - 700, ind, 'ro')
    end
end

%% Detect
%{
% Filter point cloud
xMin = 0.0;     % Minimum value along X-axis.
yMin = -39.68;  % Minimum value along Y-axis.
zMin = -5.0;    % Minimum value along Z-axis.
xMax = 69.12;   % Maximum value along X-axis.
yMax = 39.68;   % Maximum value along Y-axis.
zMax = 5.0;     % Maximum value along Z-axis.
xStep = 0.16;   % Resolution along X-axis.
yStep = 0.16;   % Resolution along Y-axis.
pointCloudRange = [xMin,xMax,yMin,yMax,zMin,zMax];

ptCloudGray = filterpc(ptCloudGray, pointCloudRange);

trainedDetector = load('/home/spencer/datasets/rob_535/ppillars/chkpts/net_checkpoint__8968__2021_11_22__16_52_50.mat');
detector = trainedDetector.net;
confidenceThreshold = 0.1;
[box,score,labels] = detect(detector,ptCloudGray,'Threshold',confidenceThreshold);
[~, idx] = max(score);
plot(cuboidModel(box(idx, :)))


function ptCloudFiltered = filterpc(ptCloud, pcRange)
    xmin = pcRange(1,1);
    xmax = pcRange(1,2);
    ymin = pcRange(1,3);
    ymax = pcRange(1,4);
    zmin = pcRange(1,5);
    zmax = pcRange(1,6);

    pos = find( ptCloud.Location(:,1) < xmax ...
                & ptCloud.Location(:,1) > xmin ...
                & ptCloud.Location(:,2) < ymax ...
                & ptCloud.Location(:,2) > ymin ...
                & ptCloud.Location(:,3) < zmax ...
                & ptCloud.Location(:,3) > zmin);    
    ptCloudFiltered = select(ptCloud, pos, 'OutputSize', 'full');
    ptCloudFiltered = removeInvalidPoints(ptCloudFiltered);
end
%}
