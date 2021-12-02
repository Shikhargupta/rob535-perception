files = dir('/home/spencer/datasets/rob_535/trainval/*/*_image.jpg');
idx = randi(numel(files));
snapshot = [files(idx).folder, '/', files(idx).name];
disp(snapshot)

xyz = read_bin(strrep(snapshot, '_image.jpg', '_cloud.bin'));
xyz = reshape(xyz, [], 3)';

change_axes = [0 0 1; -1 0 0; 0 -1 0];
xyz = change_axes * xyz;

[thetas, phis, rs] = cart2sph(xyz(1, :),xyz(2, :),xyz(3, :));
thetas = sort(rad2deg(thetas));
phis = sort(rad2deg(phis));

% Determine theta values
diffs = thetas - circshift(thetas, 1);
diffs = diffs(diffs > 0.001);
nsteps_th = length(diffs) + 1
step_th = mean(diffs)
min_th = min(thetas)
max_th = max(thetas)

% Test guess
thetas_true = -40.2 : 0.2 : 40.2;
max_diff = 0;
for i = 1:length(thetas)
    theta = thetas(i);
    min_diff = min(abs(thetas_true - theta));
    if min_diff > max_diff
        max_diff = min_diff;
    end
end
max_diff

% Determine phi values
diffs = phis - circshift(phis, 1);
diffs = diffs(diffs > 0.001);
nsteps_phi = length(diffs) + 1
step_phi = mean(diffs)
min_phi = min(phis)
max_phi = max(phis)

% Test guess
phis_true = -20 : 0.4 : 20;
max_diff = 0;
for i = 1:length(phis)
    phi = phis(i);
    min_diff = min(abs(phis_true - phi));
    if min_diff > max_diff
        max_diff = min_diff;
    end
end
max_diff

function data = read_bin(file_name)
id = fopen(file_name, 'r');
data = fread(id, inf, 'single');
fclose(id);
end
