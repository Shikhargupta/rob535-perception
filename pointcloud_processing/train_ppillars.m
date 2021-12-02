%% POSSIBLE FUTURE UPDATES
% 1. Add data augmentation
% 2. Add transfer learning from pretrained model

%% Set folders

train_in = '/home/spencer/datasets/rob_535/trainval/';
train_temp = '/home/spencer/datasets/rob_535/ppillars/trainval/';
train_chkpt = '/home/spencer/datasets/rob_535/ppillars/chkpts/';
write_pcd = true;

%% Create initial datastores

lidarData = fileDatastore(train_in,"ReadFcn",@(x) readpc(x),'IncludeSubfolders',true, 'FileExtensions', '.jpg');
gtData = fileDatastore(train_in,"ReadFcn",@(x) readbbox(x),'IncludeSubfolders',true, 'FileExtensions', '.jpg');

%% Import point clouds and ground truth

fprintf("Loading data...\n")
nfiles = length(lidarData.Files);
pcs = {};
gts = {};
tmpStr = '';
for i = 1:nfiles
    ptCloud = read(lidarData);
    gt = read(gtData);

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
    voxelSize = [xStep,yStep];

    [keep, ptCloudFiltered] = filterpc(ptCloud, gt, pointCloudRange);
    if keep
        pcs{end + 1} = ptCloudFiltered;
        gts{end + 1} = gt;
    end

    % Progress message
    msg = sprintf('Processing data %3.2f%% complete', (i/nfiles)*100.0);
    fprintf(1,'%s',[tmpStr, msg]);
    tmpStr = repmat(sprintf('\b'), 1, length(msg));
end
fprintf("Data loaded.\n")

%% Select train and test sets

rng(1); % Keep this 1 if reusing saved PCD files
shuffledIndices = randperm(length(pcs));
idx = floor(0.7 * length(shuffledIndices));

trainData = pcs(shuffledIndices(1:idx));
testData = pcs(shuffledIndices(idx + 1: end));

trainLabels = gts(shuffledIndices(1:idx));
testLabels = gts(shuffledIndices(idx + 1: end));

%% Create datastores for train data

% Write point clouds to PCD format for efficient importing
if write_pcd
    fprintf("Writing data to PCD files...")
    save2PCD(trainData, train_temp)
    fprintf("PCD files written.\n")
end

% Create point cloud datastore from PCD files
pcd = fileDatastore(train_temp,'ReadFcn',@(x) pcread(x));

% Create table of ground truth bbox data
trainTable = cuboids2table(trainLabels);

% Create boxLabelDatastore
bds = boxLabelDatastore(trainTable);

% Combine the datastores
cds = combine(pcd, bds);

%% Training

% Define number of prominent pillars.
P = 12000; 

% Define number of points per pillar.
N = 100;

% Estimate anchor boxes from training data.
anchorBoxes = calculateanchors(trainTable);
if size(anchorBoxes{1}, 2) == 2
    anchorBoxes{1} = anchorBoxes{2};
end
classNames = trainTable.Properties.VariableNames;

% Define the PointPillars detector.
detector = pointPillarsObjectDetector(pointCloudRange,classNames,anchorBoxes,...
    'VoxelSize',voxelSize,'NumPillars',P,'NumPointsPerPillar',N);

% Define the training options and hyperparameters
if canUseParallelPool
    dispatchInBackground = true;
else
    dispatchInBackground = false;
end
options = trainingOptions('adam',...
    'Plots',"none",...
    'MaxEpochs',60,...
    'MiniBatchSize',3,...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'LearnRateSchedule',"piecewise",...
    'InitialLearnRate',0.0002,...
    'LearnRateDropPeriod',15,...
    'LearnRateDropFactor',0.8,...
    'ExecutionEnvironment','gpu',...
    'DispatchInBackground',dispatchInBackground,...
    'BatchNormalizationStatistics','moving',...
    'ResetInputNormalization',false,...
    'CheckpointPath',train_chkpt);

% Run training
fprintf("Start training...\n")
[detector,info] = trainPointPillarsObjectDetector(cds,detector,options);

%% Helper functions

function ptCloudGray = readpc(snapshot)
% Import image
img = imread(snapshot);

% Import projection matrix
proj = read_bin(strrep(snapshot, '_image.jpg', '_proj.bin'));
proj = reshape(proj, [4, 3])';

% Import pointcloud data
xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
xyz = reshape(xyz, [], 3)';
rotate_axes = [0 0 1; -1 0 0; 0 -1 0];
xyz = rotate_axes * xyz;
ptCloud = pointCloud(xyz');

ptCloudGray = fuseCameraToLidar(rgb2gray(img), ... % convert to grayscale to resemble reflectance/intensity
    ptCloud, ...
    cameraIntrinsics([proj(1, 1), proj(2, 2)], [proj(1, 3), proj(2, 3)], size(img, 1, 2)), ...
    rigid3d(inv(rotate_axes), [0 0 0]));
ptCloudGray.Intensity = double(ptCloudGray.Color(:, 1));

end

function [keep, ptCloudFiltered] = filterpc(ptCloud, gt, pcRange)
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

    idx = findPointsInsideCuboid(gt.model, ptCloudFiltered);
    keep = length(idx) > 50;
end

function gt = readbbox(snapshot)

bbox = read_bin(strrep(snapshot, '_image.jpg', '_bbox.bin'));
bbox = reshape(bbox, 11, [])';
rotate_axes = [0 0 1; -1 0 0; 0 -1 0];
gt.model = getcuboid(bbox, rotate_axes);

classes = readmatrix('/home/spencer/datasets/rob_535/classes.csv');
gt.label = classes(int64(bbox(10)) + 1, 3); 

end

function save2PCD(pcs, outdir)
    if ~exist(outdir, 'dir')
        mkdir(outdir)
    end
    numFiles = length(pcs);
    tmpStr = '';
    for i = 1:numFiles
        pc = pcs{i};
        pcFilePath = fullfile(outdir, sprintf('%06d.pcd', i));
        pcwrite(pc, pcFilePath);

        % Progress message
        msg = sprintf('Processing data %3.2f%% complete', (i/numFiles)*100.0);
        fprintf(1,'%s',[tmpStr, msg]);
        tmpStr = repmat(sprintf('\b'), 1, length(msg));
    end
end

function gt_table = cuboids2table(gts)
    nfiles = length(gts);
    gt_table_data = cell(nfiles, 3);
    for i = 1:nfiles
        gt = gts{i};
        label_idx = gt.label + 1;
        cuboid_model = gt.model;
        gt_table_data{i, label_idx} = cuboid_model.Parameters;
    end
    gt_table = cell2table(gt_table_data);
    for i = 1:3
        gt_table.Properties.VariableNames{i} = strcat('label_', num2str(i - 1));
    end
end

function anchors = calculateanchors(labels)
    anchors = [];
    classNames = labels.Properties.VariableNames;
    
    % Calculate the anchors for each class label.
    for ii = 1:numel(classNames)
        bboxCells = table2array(labels(:,ii));
        lwhValues = [];
        
        % Accumulate the lengths, widths, heights from the ground truth
        % labels.
        for i = 1 : height(bboxCells)
            if(~isempty(bboxCells{i}))
                lwhValues = [lwhValues; bboxCells{i}(:, 4:6)];
            end
        end
        
        % Calculate the mean for each. 
        meanVal = mean(lwhValues, 1);
        
        % With the obtained mean values, create two anchors with two 
        % yaw angles, 0 and 90.
%         classAnchors = [{num2cell([meanVal, -1.78, 0])}, {num2cell([meanVal, -1.78, pi/2])}];
        classAnchors = [[meanVal, -1.78, 0]; [meanVal, -1.78, pi/2]];
        
        anchors = [anchors; {classAnchors}];
    end
end

