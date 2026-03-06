function [x,y] = fDnnAxisClbEach(camClb,trjClb,x,y,indx,accp)
%% return x axis info for bella magspec cameras
%
% [x,y] = fBellaXAxisV01(camClb,trjClb)
%
% x,y: structure contains x,y axis info
%   .mm: x (or z) [mm]
%   .dx: dx
%   .incAgl: incident angle to screen [deg]
%   .path: path length [m]
%   .divFX: divergin factor for X plane
%   .divFY: diverging factor for Y plane
%   .accp: acceptance (half angle) [mrad]
%   .mmt: normalized momentum [MeV/cB]
%   .dp: dp for each pixel [MeV/cB]
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
% trjCalib: trajectory calibration in structure
%   .mmt: momentum [MeV/c]
%   .screen: screen position [mm]
%   .incAgl: electron incident angle to screen [deg]
%   .path: total path length [m]
%   .divFX: diverging factor for x
%   .fivFY: diverging factor for y
%

%% Written by Kei Nakamura
% 2018/2/9 ver.1: created
%

%% x axis

% pixel
xx = 1:camClb.width;    % save ROI
x(indx).pixel = xx(camClb.xSt:camClb.xEd); % analysis ROI

% x or z, dx
dx = camClb.fov/camClb.width;   %dx, fov/pixel#
xEnd = camClb.leftPos - camClb.fov + dx;  % right edge
xx = camClb.leftPos:-dx:xEnd;    % x (or z) in mm
x(indx).mm = xx(camClb.xSt:camClb.xEd);
dx = abs(dx);
x(indx).dx = dx;

% electron incident angle to screen
x(indx).incAgl = interp1(trjClb.screen,trjClb.incAgl,x(indx).mm,'PCHIP');

% path length [m]
x(indx).path = interp1(trjClb.screen,trjClb.path,x(indx).mm,'PCHIP');

% divergin factor for X plane
%x(indx).divFX = interp1(trjClb.screen,trjClb.divFX,x(indx).mm,'cubic');

% diverging factor for Y plane
x(indx).divFY = interp1(trjClb.screen,trjClb.divFY,x(indx).mm,'PCHIP');

% acceptance (half angle, screen)
x(indx).accp = 0.5*accp./(x(indx).path.*x(indx).divFY);      % clearance 33 (40) mm

% normalized momentum (1T), dp
x(indx).mmt = interp1(trjClb.screen,trjClb.mmt,x(indx).mm,'spline');
diff1 = diff(x(indx).mmt);
%diff1 = fSmoothAryV01(diff(x(indx).mmt),9);
x(indx).dp = 0.5*([diff1,diff1(end)] + [diff1(1),diff1]);

%% y axis

% pixel
yy = 1:camClb.height;   % save ROI
y(indx).pixel = yy(camClb.ySt:camClb.yEd);    % analysis ROI

% dy, y
y(indx).dy = dx;
y(indx).mm = y(indx).pixel*dx;
y(indx).mm(:) = y(indx).mm(:) - dx*camClb.yCntr; % zero at yCntr pixel


% figure(6)
% plot(trjClb.screen,trjClb.incAgl,'ro')
% hold on
% plot(x.mm,x.incAgl, 'b-')
% hold off
