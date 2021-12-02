function [img, xyz, proj, bbox] = import_snapshot_random()
% Choose random snapshot
files = dir('/home/spencer/datasets/rob_535/trainval/*/*_image.jpg');
idx = randi(numel(files));
snapshot = [files(idx).folder, '/', files(idx).name];
%snapshot = '/home/spencer/datasets/rob_535/test/72e86196-dd16-4d1a-8285-26057e0aea6a/0054_image.jpg';
disp(snapshot)

% Import image
img = imread(snapshot);

% Import pointcloud data
xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
xyz = reshape(xyz, [], 3)';

% Import projection matrix
proj = read_bin(strrep(snapshot, '_image.jpg', '_proj.bin'));
proj = reshape(proj, [4, 3])';

% Import bounding box and class
bbox = read_bin(strrep(snapshot, '_image.jpg', '_bbox.bin'));
bbox = reshape(bbox, 11, [])';
end