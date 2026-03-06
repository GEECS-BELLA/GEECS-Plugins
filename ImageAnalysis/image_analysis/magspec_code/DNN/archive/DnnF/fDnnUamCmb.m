function im1 = fDnnUamCmb(imgC,anglC,dAnglC,x,xA,yA)
%% return uniform angle and momentum, combined images for dnn magspec cameras
%
% [im1,im2,im3] = fBellaUamCmbV01(imgC,anglC,dAnglC,x,xA,yA)
%
% im: angle and momentum uniform images
%
% imgC: image cell
% anglC: angle Cell
% dAngleC: dAngle cell
% x: x axis info for each image
% xA: x axix info, stiched
% yA: structure with y info
%   .angl: angle [mrad]

%% Written by Kei Nakamura
% 2018/2/22 ver.1: created
%

%% main body

img1 = fBellaUaV01(imgC{1},anglC{1},dAnglC{1},yA.angl);
img1 = fBellaMmtBinV01(img1,x(1).bin);

img2 = fBellaUaV01(imgC{2},anglC{2},dAnglC{2},yA.angl);
img2 = fBellaMmtBinV01(img2,x(2).bin);

img3 = fBellaUaV01(imgC{3},anglC{3},dAnglC{3},yA.angl);
img3 = fBellaMmtBinV01(img3,x(3).bin);

img4 = fBellaUaV01(imgC{4},anglC{4},dAnglC{4},yA.angl);
img4 = fBellaMmtBinV01(img4,x(4).bin);

img2(:,1) = 0; img3(:,end) = 0;   % put 0 for boundaries

img1 = [img4 img3 img2 img1];
dpB = [x(4).dpB x(3).dpB x(2).dpB x(1).dpB];  % combined dp
mmtB = [x(4).mmtB x(3).mmtB x(2).mmtB x(1).mmtB];   % combined mmt

im1 = fBellaUmV01(img1,mmtB,dpB,xA(1));    % uniform dp, [fC]
im1(:,1)=0;
