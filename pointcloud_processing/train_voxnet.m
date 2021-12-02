%% POSSIBLE IMPROVEMENTS
% - Rotate bbox to be parallel with xy plane (in practice estimate ground
% plane surface normal)

%% Parameters

pts_thresh = 300;
dropped_classes = [0, 14, 15, 16, 17, 19, 20, 22]; % < 20 in train
ntrain = 7000;

%% Set folders

train_in = '/home/spencer/datasets/rob_535/trainval/';
train_chkpt = '/home/spencer/datasets/rob_535/voxnet/chkpts_big/';
write_pcd = true;

%% Create initial datastores

% No built in deep copy so create multiple instances
pcDS = fileDatastore(train_in,"ReadFcn",@(x) readpc(x, pts_thresh, dropped_classes),'IncludeSubfolders',true, 'FileExtensions', '.jpg');
pcDSTrain = fileDatastore(train_in,"ReadFcn",@(x) readpc(x, pts_thresh, dropped_classes),'IncludeSubfolders',true, 'FileExtensions', '.jpg');
pcDSVal = fileDatastore(train_in,"ReadFcn",@(x) readpc(x, pts_thresh, dropped_classes),'IncludeSubfolders',true, 'FileExtensions', '.jpg');

%% Import point clouds and ground truth data

fprintf("Loading train data...\n")
nfiles = length(pcDS.Files);
valid_files = [];
tmpStr = '';
for i = 1:ntrain
    if ~isempty(read(pcDS))
        valid_files = [valid_files, i];
    end

    % Progress message
    msg = sprintf('Processing data %3.2f%% complete', (i/ntrain)*100.0);
    fprintf(1,'%s',[tmpStr, msg]);
    tmpStr = repmat(sprintf('\b'), 1, length(msg));
end
pcDSTrain.Files = pcDSTrain.Files(valid_files);
fprintf("Train data loaded.\n")

fprintf("\nLoading validation data...\n")
valid_files = [];
tmpStr = '';
for i = ntrain + 1:nfiles
    if ~isempty(read(pcDS))
        valid_files = [valid_files, i];
    end

    % Progress message
    msg = sprintf('Processing data %3.2f%% complete', ((i - ntrain)/(nfiles - ntrain + 1))*100.0);
    fprintf(1,'%s',[tmpStr, msg]);
    tmpStr = repmat(sprintf('\b'), 1, length(msg));
end
pcDSVal.Files = pcDSVal.Files(valid_files);
fprintf("\nValidation data loaded.\n")

%% Plot label distributions

labelDSTrain = transform(pcDSTrain, @(data) data{2});
labels = readall(labelDSTrain);
figure
histogram(labels)
title("Train")

labelDSVal = transform(pcDSVal, @(data) data{2});
labels = readall(labelDSVal);
figure
histogram(labels)
title("Val")

%% Augmentation

pcDSTrain = transform(pcDSTrain, @augmentPointCloudData);

dataOut = preview(pcDSTrain);
figure
pcshow(dataOut{1});
title(dataOut{2});

%% Voxelize

pcDSTrain = transform(pcDSTrain, @formOccupancyGrid);
pcDSVal = transform(pcDSVal, @formOccupancyGrid);

%% Confirm working

data = preview(pcDSTrain);
figure
p = patch(isosurface(data{1},0.5));
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1 1 1])
view(45,45)
camlight; 
lighting phong
title(data{2});

%% Define network architecture

layers = [image3dInputLayer([32 32 32],'Name','inputLayer','Normalization','none'),...
    convolution3dLayer(5,32,'Stride',2,'Name','Conv1'),...
    leakyReluLayer(0.1,'Name','leakyRelu1'),...
    convolution3dLayer(3,32,'Stride',1,'Name','Conv2'),...
    leakyReluLayer(0.1,'Name','leakyRulu2'),...
    maxPooling3dLayer(2,'Stride',2,'Name','maxPool'),...
    fullyConnectedLayer(128,'Name','fc1'),...
    reluLayer('Name','relu'),...
    dropoutLayer(0.5,'Name','dropout1'),...
    fullyConnectedLayer(15,'Name','fc2'),...
    softmaxLayer('Name','softmax'),...
    classificationLayer('Name','crossEntropyLoss')];

voxnet = layerGraph(layers);

%% Training options

miniBatchSize = 32;
dsLength = length(pcDSTrain.UnderlyingDatastore.Files);
iterationsPerEpoch = floor(dsLength/miniBatchSize);
dropPeriod = floor(8000/iterationsPerEpoch);

options = trainingOptions('sgdm','InitialLearnRate',0.01,'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule','Piecewise',...
    'LearnRateDropPeriod',dropPeriod,...
    'ValidationData',pcDSVal,'MaxEpochs',60,...
    'DispatchInBackground',false,...
    'Shuffle','every-epoch', ...
    'CheckpointPath',train_chkpt);

%% Train network

voxnet = trainNetwork(pcDSTrain,voxnet,options);

%% Evaluate

valLabelSet = transform(pcDSVal,@(data) data{2});
valLabels = readall(valLabelSet);
outputLabels = classify(voxnet,pcDSVal);
accuracy = nnz(outputLabels == valLabels) / numel(outputLabels);
disp(accuracy)

confusionchart(valLabels,outputLabels)

%% Helper functions

function dataOut = readpc(snapshot, pts_thresh, dropped_classes)
    % Classes list
    classes = ["Unknown", "Compacts", "Sedans", "SUVs", "Coupes", ...
        "Muscle", "SportsClassics", "Sports", "Super", "Motorcycles", ...
        "OffRoad", "Industrial", "Utility", "Vans", "Cycles", ...
        "Boats", "Helicopters", "Planes", "Service", "Emergency", ...
        "Military", "Commercial", "Trains"];
    remaining_classes = classes(setdiff(1:length(classes), dropped_classes + 1));
    
    % Import pointcloud data
    xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
    xyz = reshape(xyz, [], 3)';
    rotate_axes = [0 0 1; -1 0 0; 0 -1 0];
    xyz = rotate_axes * xyz;
    ptCloud = pointCloud(xyz');

    % Import ground truth data
    bbox = read_bin(strrep(snapshot, '_image.jpg', '_bbox.bin'));
    bbox = reshape(bbox, 11, [])';
    rotate_axes = [0 0 1; -1 0 0; 0 -1 0];
    model = getcuboid(bbox, rotate_axes);
    
    % Set output data if the class is not dropped and there are enough pts
    class_id = int64(bbox(10));
    bbox_pts = findPointsInsideCuboid(model, ptCloud);
    dataOut = {};
    if sum(dropped_classes == class_id) == 0 && ...
            length(bbox_pts) > pts_thresh
        bboxPtCloud = select(ptCloud, bbox_pts);
        label = categorical(classes(class_id + 1), remaining_classes);
        dataOut = {bboxPtCloud, label};
    end
end

function dataOut = augmentPointCloudData(data)
    ptCloud = data{1};
    label = data{2};
    
    % Apply randomized rotation about Z axis.
    tform = randomAffine3d('Rotation',@() deal([0 0 1],360*rand),'Scale',[0.98,1.02],'XReflection',true,'YReflection',true); % Randomized rotation about z axis
    ptCloud = pctransform(ptCloud,tform);
    
    % Apply jitter to each point in point cloud
    amountOfJitter = 0.01;
    numPoints = size(ptCloud.Location,1);
    D = zeros(size(ptCloud.Location),'like',ptCloud.Location);
    D(:,1) = diff(ptCloud.XLimits)*rand(numPoints,1);
    D(:,2) = diff(ptCloud.YLimits)*rand(numPoints,1);
    D(:,3) = diff(ptCloud.ZLimits)*rand(numPoints,1);
    D = amountOfJitter.*D;
    ptCloud = pctransform(ptCloud,D);
    
    dataOut = {ptCloud,label};
end

function dataOut = formOccupancyGrid(data)
    grid = pcbin(data{1},[32 32 32]);
    occupancyGrid = zeros(size(grid),'single');
    for ii = 1:numel(grid)
        occupancyGrid(ii) = ~isempty(grid{ii});
    end
    label = data{2};
    dataOut = {occupancyGrid,label};
end