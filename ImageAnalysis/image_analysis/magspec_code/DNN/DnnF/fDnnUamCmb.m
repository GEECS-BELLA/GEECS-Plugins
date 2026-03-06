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

% figure(101)
% subplot(1,4,4)
% pcolor(img1)
% shading interp
% colorbar
% subplot(1,4,3)
% pcolor(img2)
% shading interp
% colorbar
% subplot(1,4,2)
% pcolor(img3)
% shading interp
% colorbar
% subplot(1,4,1)
% pcolor(img4)
% shading interp
% colorbar

img1 = [img4 img3 img2 img1];

% figure(101)
% subplot(1,4,1)
% pcolor(img1)
% shading interp
% colorbar
% title('conbined')
% axis([1 200 0 120])

dpB = [x(4).dpB x(3).dpB x(2).dpB x(1).dpB];  % combined dp
mmtB = [x(4).mmtB x(3).mmtB x(2).mmtB x(1).mmtB];   % combined mmt

im1 = fBellaUmV01(img1,mmtB,dpB,xA(1));    % uniform dp, [fC]
im1(:,1)=0;

% subplot(1,4,4)
% pcolor(im1)
% shading interp
% colorbar
% title('de uniform')
% axis([1 200 0 120])
%
% figure(102)
% subplot(1,2,1)
% plot(dpB)
% axis([1 200 0.4 1])
% title('dE')
% subplot(1,2,2)
% plot(mmtB,'o')
% axis([70 110 0 200])
% title('E')
%
% figure(103)
% subplot(1,2,1)
% plot(x(4).dpB,'o')
% %axis([1 200 0.4 1])
% title('dE-cam4')
% subplot(1,2,2)
% plot(x(3).dpB,'o')
% %axis([1 200 0 200])
% title('dE-cam3')
