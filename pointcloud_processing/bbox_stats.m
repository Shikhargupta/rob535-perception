
train_in = '/home/spencer/datasets/rob_535/trainval/';
lidarData = fileDatastore(train_in,"ReadFcn",@(x) readpc(x),'IncludeSubfolders',true, 'FileExtensions', '.jpg');
gtData = fileDatastore(train_in,"ReadFcn",@(x) readbbox(x),'IncludeSubfolders',true, 'FileExtensions', '.jpg');

fprintf("Loading data...\n")
nfiles = length(lidarData.Files);
stats_map = containers.Map;
n_pts = zeros(22, nfiles);
tmpStr = '';
for i = 1:nfiles
    gt = read(gtData);
    if isKey(stats_map, num2str(gt.label))
        stats_map(num2str(gt.label)) = [stats_map(num2str(gt.label)), (gt.model.Dimensions)'];
    else
        stats_map(num2str(gt.label)) = (gt.model.Dimensions)';
    end

    ptCloud = read(lidarData);
    gtPts = findPointsInsideCuboid(gt.model, ptCloud);
    n_pts(gt.label, i) = length(gtPts);
    %if length(gtPts) > 50
    %    gtPtCloud = select(ptCloud, gtPts);
    %end

    % Progress message
    msg = sprintf('Processing data %3.2f%% complete', (i/nfiles)*100.0);
    fprintf(1,'%s',[tmpStr, msg]);
    tmpStr = repmat(sprintf('\b'), 1, length(msg));
end
fprintf("Data loaded.\n")

maxdims = zeros(22, 4);
mindims = zeros(22, 4);
vardims = zeros(22, 4);
count = zeros(22, 2);
for i = 0:22
    if isKey(stats_map, num2str(i))
        stats.vals = stats_map(num2str(i));
        stats.min = zeros(3, 1);
        stats.max = zeros(3, 1);
        stats.var = zeros(3, 1);
        for j = 1:3
            stats.min(j) = min(stats.vals(j, :));
            stats.max(j) = max(stats.vals(j, :));
            stats.var(j) = var(stats.vals(j, :));
        end
        stats_map(num2str(i)) = stats;

        maxdims(i, :) = [i; stats.max];
        mindims(i, :) = [i; stats.min];
        vardims(i, :) = [i; stats.var];
        count(i, :) = [i, numel(stats.vals(j, :))];
    end
end

%% Helper functions

function ptCloud = readpc(snapshot)

xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
xyz = reshape(xyz, [], 3)';
rotate_axes = [0 0 1; -1 0 0; 0 -1 0];
xyz = rotate_axes * xyz;
ptCloud = pointCloud(xyz');

end

function gt = readbbox(snapshot)

bbox = read_bin(strrep(snapshot, '_image.jpg', '_bbox.bin'));
bbox = reshape(bbox, 11, [])';
rotate_axes = [0 0 1; -1 0 0; 0 -1 0];
gt.model = getcuboid(bbox, rotate_axes);

classes = readmatrix('/home/spencer/datasets/rob_535/classes.csv');
gt.label = classes(int64(bbox(10)) + 1, 1); 

end


