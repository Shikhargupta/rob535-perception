function [ptCloudOrg] = xyz2organizedpc(xyz)
% Rotate such that x points forward, y points left, and z points up
rotate_axes = [0 0 1; -1 0 0; 0 -1 0];
xyz = rotate_axes * xyz;

% True lidar specs
% - phis = -20 : 0.4 : 20 (in degrees)
% - thetas = -40.2 : 0.2 : 40.2 (in degrees)

verticalFoV = [20, -20];
verticalResolution = length(-20 : 0.4 : 20);
horizontalResolution = 360 / 0.2 + 1;

params = lidarParameters(verticalResolution, verticalFoV,...
    horizontalResolution);

ptCloud = pointCloud(xyz');
ptCloudOrg = pcorganize(ptCloud, params);

end

