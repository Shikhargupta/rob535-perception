train_in = '/home/spencer/datasets/rob_535/trainval/';
train_chkpt = '/home/spencer/datasets/rob_535/voxnet/chkpts_big/';

old_classes = categorical(["Unknown", "Compacts", "Sedans", "SUVs", "Coupes", ...
    "Muscle", "SportsClassics", "Sports", "Super", "Motorcycles", ...
    "OffRoad", "Industrial", "Utility", "Vans", "Cycles", ...
    "Boats", "Helicopters", "Planes", "Service", "Emergency", ...
    "Military", "Commercial", "Trains"]);
new_classes = categorical(["Label1", "Label2", "Label0"]);

nn = [train_chkpt, 'net_checkpoint__2160__2021_12_01__17_00_11.mat'];
nn = load(nn);
nn = nn.net;

pcDS = fileDatastore(train_in,"ReadFcn",@(x) readpc(x),'IncludeSubfolders',true, 'FileExtensions', '.jpg');

fprintf("Testing...\n")
nfiles = length(pcDS.Files);
tmpStr = '';
gtLabels = [];
predLabels = [];
scores = [];
for i = 1:nfiles
    data = read(pcDS);
    ptCloud = data{1};
    gtLabel = convertClass(data{2}, old_classes, new_classes);
    gtLabels = [gtLabels, gtLabel];

    % Segment the ground
    groundPtsIdx = segmentGroundSMRF(ptCloud, 'ElevationThreshold', 0.2);
    
    % Extract and plot the ground points
    groundPtCloud = select(ptCloud, groundPtsIdx);
        
    % Extract non-ground points, retaining organized structure
    nonGroundPtCloud = select(ptCloud, ~groundPtsIdx, 'OutputSize', 'full');
    
    % Segment non-ground points
    distThreshold = 1; % in meters
    [labels, numClusters] = segmentLidarData(nonGroundPtCloud, distThreshold);
    
    % Find point clouds with enough points
    count = zeros(numClusters, 1);
    for j = 1:numClusters
        count(j) = sum(sum(labels == j));
    end
    lidx = find(count > 300);

    % Make prediction
    max_score = 0.95; % Guess Label1 unless really confident
    predLabel = categorical("Label1");
    for j = 1:length(lidx)
        labelpc = select(nonGroundPtCloud, labels==lidx(j));
        model = pcfitcuboid(labelpc);
        if model.Dimensions(3) < 5  && ... 
            (model.Dimensions(2) < 4 || model.Dimensions(1) < 4) && ...
            sum(model.Dimensions > 14) == 0
            
            grid = formOccupancyGrid(labelpc);
            [ypred, class_scores] = classify(nn, grid);
            if max(class_scores) > max_score
                predLabel = ypred;
                predLabel = convertClass(predLabel, old_classes, new_classes);
                max_score = max(class_scores);
                %pcshow(labelpc)
                %waitforbuttonpress;
            end
        end
    end
    scores = [scores, max_score];
    predLabels = [predLabels, predLabel];

    % Progress message
    msg = sprintf('Processing data %3.2f%% complete', (i/nfiles)*100.0);
    fprintf(1,'%s',[tmpStr, msg]);
    tmpStr = repmat(sprintf('\b'), 1, length(msg));
end
fprintf("Done.\n")

%% Helper functions

function dataOut = readpc(snapshot)
    % Classes list
    classes = ["Unknown", "Compacts", "Sedans", "SUVs", "Coupes", ...
        "Muscle", "SportsClassics", "Sports", "Super", "Motorcycles", ...
        "OffRoad", "Industrial", "Utility", "Vans", "Cycles", ...
        "Boats", "Helicopters", "Planes", "Service", "Emergency", ...
        "Military", "Commercial", "Trains"];
    
    % Import pointcloud data
    xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
    xyz = reshape(xyz, [], 3)';
    ptCloud = xyz2organizedpc(xyz);

    % Import ground truth data
    bbox = read_bin(strrep(snapshot, '_image.jpg', '_bbox.bin'));
    bbox = reshape(bbox, 11, [])';

    % Set output data if the class is not dropped and there are enough pts
    class_id = int64(bbox(10));
    class_label = classes(class_id + 1);
    label = categorical(class_label, classes);
    dataOut = {ptCloud, label};
end

function occupancyGrid = formOccupancyGrid(ptCloud)
    grid = pcbin(ptCloud,[32 32 32]);
    occupancyGrid = zeros(size(grid),'single');
    for ii = 1:numel(grid)
        occupancyGrid(ii) = ~isempty(grid{ii});
    end
end

function newclass = convertClass(oldclass, oldclasses, newclasses)
    id = find(oldclasses == oldclass);
    if id < 9
        newclass = categorical(newclasses(1));
    elseif id > 8 && id < 15
        newclass = categorical(newclasses(2));
    else
        newclass = categorical(newclasses(3));
    end
end
