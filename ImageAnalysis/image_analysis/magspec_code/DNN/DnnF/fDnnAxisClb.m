function [x,y] = fDnnAxisClb(camClb,trjFCalib,trjSCalib,accp)
%% return x axis cell for bella magspec cameras
%
% [x,y] = fBellaAxisAllV01(camClb,trjFCalib,trjSCalib)
%
% x,y: structure contains x,y axis info
%   .mm: x (or z, y) [mm]
%   .dx: dx (or dy) [mm]
%   .incAgl: incident angle to screen [deg]
%   .path: path length [m]
%   .divFX: divergin factor for X plane
%   .divFY: diverging factor for Y plane
%   .accp: acceptance (half angle) [mrad]
%
% camClb: structure of calibration for bella magspec cameras
%   .fov: field of view [mm]
%   .yOffset: y offset for roi [pixel]
%   .height: height of image [pixel]
%   .xOffset: x offset for roi [pixel]
%   .width: width of image [pixel]
%   .leftPos: left position of image [mm]
%   .yCnter: y center pixel
%   .ySt,yEd: y start and end for analysis ROI
%   .xSt,xEd: x start and end for analysis ROI
%
% trjFCalib,trjSCalib: trajectory front and side calibration in structure
%   .mmt: momentum [MeV/c]
%   .screen: screen position [mm]
%   .incAgl: electron incident angle to screen [deg]
%   .path: total path length [m]
%   .divFX: diverging factor for x
%   .divFY: diverging factor for y
%

%% Written by Kei Nakamura
% 2018/2/9 ver.1: created
%

%% main body

% allocation
x(4).pixel = zeros(1,camClb(4).width);
y(4).pixel = zeros(1,camClb(4).height);

% front screen
for ii = 1:2
    [x,y] = fDnnAxisClbEach(camClb(ii),trjFCalib,x,y,ii,accp(1));
end

% bottom screen
for ii = 3:4
    [x,y] = fDnnAxisClbEach(camClb(ii),trjSCalib,x,y,ii,accp(2));
    %x(ii).dp = -x(ii).dp;
end
% cam3 compensation

y(3).dy = 0.94*y(3).dy;
y(3).mm = 0.94*y(3).mm;
