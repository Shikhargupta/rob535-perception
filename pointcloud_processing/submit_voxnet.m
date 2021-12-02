test_in = '/home/spencer/datasets/rob_535/test/';
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

files = dir('/home/spencer/datasets/rob_535/test/*/*_image.jpg');
f = fopen('voxent_submission.txt', 'w');
fprintf(f, 'guid/image,label\n');

fprintf("Testing...\n")
tmpStr = '';
predLabels = [];
scores = [];
nfiles = numel(files);
for i = 1:nfiles
    ptCloud = readpc([files(i).folder, '/', files(i).name]);

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
    predLabel = '1';
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
    fprintf(f, ...
        [files(i).folder(37:end), '/',files(i).name(1:end - 10), ',', predLabel, '\n']);

    % Progress message
    msg = sprintf('Processing data %3.2f%% complete', (i/nfiles)*100.0);
    fprintf(1,'%s',[tmpStr, msg]);
    tmpStr = repmat(sprintf('\b'), 1, length(msg));
end
fprintf("Done.\n")
fclose(f);
%% Helper functions

function ptCloud = readpc(snapshot)
    % Import pointcloud data
    xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
    xyz = reshape(xyz, [], 3)';
    ptCloud = xyz2organizedpc(xyz);
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
        newclass = '1';
    elseif id > 8 && id < 15
        newclass = '2';
    else
        newclass = '0';
    end
end
