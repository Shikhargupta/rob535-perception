%pretrainedDetector = load('pretrainedPointPillarsDetector.mat','detector');
%detector = pretrainedDetector.detector;
close all
train_chkpt = '/home/spencer/datasets/rob_535/voxnet/chkpts/';
%train_chkpt = '/home/spencer/datasets/rob_535/voxnet/chkpts_big/';
nn = [train_chkpt, 'net_checkpoint__8520__2021_12_01__03_56_50.mat'];
%nn = [train_chkpt, 'net_checkpoint__2160__2021_12_01__17_00_11.mat'];
%nn = [train_chkpt, 'net_checkpoint__568__2021_12_01__13_56_46.mat'];
nn = load(nn);
nn = nn.net;

% Only proceed if there are more than 100 points on the vehicle
rotate_axes = [0 0 1; -1 0 0; 0 -1 0];
[img, xyz, proj, bbox] = import_snapshot_random();
%[img, xyz, proj, bbox] = import_specific_snapshot(104);
ptCloud = xyz2organizedpc(xyz);
while(length(findPointsInsideCuboid(getcuboid(bbox, rotate_axes), ptCloud)) < 300 )%|| bbox(10) < 9)
    [img, xyz, proj, bbox] = import_snapshot_random();
    ptCloud = xyz2organizedpc(xyz);
end
length(findPointsInsideCuboid(getcuboid(bbox, rotate_axes), ptCloud))
imshow(img)
set(gcf, 'position', [100, 100, 800, 600])

%confidenceThreshold = 0.5;
%[box,score,labels] = detect(detector,ptCloud,'Threshold',confidenceThreshold);

% Segment the ground
groundPtsIdx = segmentGroundSMRF(ptCloud, 'ElevationThreshold', 0.2);

% Extract and plot the ground points
groundPtCloud = select(ptCloud, groundPtsIdx);

figure
pcshow(groundPtCloud)

% Extract non-ground points, retaining organized structure
nonGroundPtCloud = select(ptCloud, ~groundPtsIdx, 'OutputSize', 'full');

% Segment non-ground points
distThreshold = 1; % in meters
[labels, numClusters] = segmentLidarData(nonGroundPtCloud, distThreshold);

% Plot the labeled results
figure(3)
colormap(hsv(numClusters))
pcshow(nonGroundPtCloud.Location, labels)
title('Point Cloud Clusters')

count = zeros(numClusters, 1);
for i = 1:numClusters
    count(i) = sum(sum(labels == i));
end
lidx = find(count > 300);
count = count(count > 300);
figure(4)
colormap(hsv(length(lidx)))
title('Point Cloud Clusters')
hold on
for i = 1:length(lidx)
    labelpc = select(nonGroundPtCloud, labels==lidx(i));
    model = pcfitcuboid(labelpc);
    if model.Dimensions(3) < 5  && ... 
        (model.Dimensions(2) < 4 || model.Dimensions(1) < 4) && ...
        sum(model.Dimensions > 14) == 0
        
        grid = formOccupancyGrid(labelpc);
        [ypred, scores] = classify(nn, grid)
        max(scores)
        count(i)

        model.Dimensions
        figure(3)
        plot(model);
        figure(4)
        pcshow(labelpc)
        waitforbuttonpress;
    end
end

function occupancyGrid = formOccupancyGrid(ptCloud)
    grid = pcbin(ptCloud,[32 32 32]);
    occupancyGrid = zeros(size(grid),'single');
    for ii = 1:numel(grid)
        occupancyGrid(ii) = ~isempty(grid{ii});
    end
end
