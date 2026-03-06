function [xA,x] = fDnnUmX(x,mmtRsl,camClb)
%% return almost uniform momentum x axis for bella magspec cameras (bin info)
%
% xA = fBellaUmXV01(x,mmtRsl)
%
% xA: almost uniform momentum x axis structure
%
% x: x axis structure
% mmtRsl: momentum resolutions
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
%   .rot: rotation [deg]

%% Written by Kei Nakamura
% 2013/3/25 ver.1: created

%% main body

%% window 1
% momentum
maxP1 = x(1).mmt(end);      % max momentum
minP1 = x(4).mmt(1);        % min momentum
xA.dp = (maxP1 - minP1)/(mmtRsl-1);  % dp for normalized
xA.mmt = [minP1:xA(1).dp:maxP1];

% acceptance
xMmt = [x(4).mmt x(3).mmt x(2).mmt x(1).mmt];
xAccp = [x(4).accp x(3).accp x(2).accp x(1).accp];
xA.accp = interp1(xMmt,xAccp,xA.mmt);

% incident angle (for solid angle estimate)
%incA = [x(2).incAgl x(1).incAgl];
%xA(1).incAgl = interp1(xMmt,incA,xA(1).mmt);

%% binning information
for indx=1:4   % loop for camera

    j = 1;  % index for binned box
    width = camClb(indx).xEd-camClb(indx).xSt+1;  % camera pixel width
    dpA = zeros(1,width);   % dp array, init
    binA = zeros(1,width);  % bib array, init

    for i=1:width % pixel loop
        dpA(j) = dpA(j) + x(indx).dp(i);
        binA(j) = binA(j) + 1;
        if dpA(j)/xA.dp > 1
            if binA(j)>1
                dpA(j) = dpA(j) - x(indx).dp(i);
                dpA(j+1) = x(indx).dp(i);
                binA(j) = binA(j) - 1;
                binA(j+1) = 1;
            end
            j = j+1;
        end
    end
    pp = find(binA==0,1,'first');
    x(indx).dpB = dpA(1:pp-1); % dp binned
    x(indx).bin = binA(1:pp-1);    % pixel# in each bin

    i = 1;
    for k=1:pp-1
        x(indx).mmtB(k) = mean(x(indx).mmt(i:i+x(indx).bin(k)-1));   % momentum binned
        i = i + x(indx).bin(k);
    end
end
