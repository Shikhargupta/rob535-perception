function [img, xyz, proj, bbox] = import_specific_snapshot(idx)
% Choose random snapshot
files = dir('/home/spencer/datasets/rob_535/trainval/*/*_image.jpg');
snapshot = [files(idx).folder, '/', files(idx).name];
%snapshot = '/home/spencer/datasets/rob_535/trainval/6d179946-a669-4f32-8438-145a0f80886c/0019_image.jpg';
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