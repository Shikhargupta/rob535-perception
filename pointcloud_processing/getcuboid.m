function model = getcuboid(bbox, Ra)
params(1:3) = Ra * bbox(4:6)';
params(4:6) = bbox(7:9);
R = rot(bbox(1:3));
R = Ra * R;
[rotx, roty, rotz] = r2e(R);
params(7:9) = [rad2deg(rotx), rad2deg(roty), rad2deg(rotz)];
model = cuboidModel(params);
end

function R = rot(n)
theta = norm(n, 2);
if theta
    n = n / theta;
    K = [0, -n(3), n(2); n(3), 0, -n(1); -n(2), n(1), 0];
    R = eye(3) + sin(theta) * K + (1 - cos(theta)) * K^2;
else
    R = eye(3);
end
end

function [rotx, roty, rotz] = r2e(R)
rotz = atan2(R(2,1),R(1,1));
roty = -asin(R(3,1));
rotx = atan2(R(3,2),R(3,3));
end