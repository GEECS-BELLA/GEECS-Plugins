function [symLeft,mx] = fFrogSSImgPrcV03(frgT,frgL,img,cmrPara,osPrp,varargin)
%% Frog single shot image processing
% [timeHlf,binWvl,symLeft] = fFrogSSImgPrcV01(frgT,frgL,img,cmrPara,lmb0,varargin)
%
% timeHlf: time (half) [fs]
% binWvl: binned wavelength [nm]
% symLeft: symmetrized left half image
%
% frgT: time axis [fs]
% frgL: wavelength axis [nm]
% img: frog image
% cmrPara
%   .binNmb:    bin #
%   .wvlRoi:    wavelength roi
%   .rotAgl:    rotation angle [deg]
%	.anlThre:   analysis threshold for camera
%	.timeOS:    time offset for image
% osPrp
%   .lmb0:      central wavelength
%   .fwhm:      fwhm
%   .peak:      peak wavelength
%   .wvl10p:    wavelengths for 10% intensity
%   .indx10p:   index for 10% intensity for eS
% varagin {1}: figure handle for figure on
%

%% Written by Kei Nakamura
% 2014/8/1 ver.1: created
% 2014/11/18 ver.2: for 3x3 binned image
% 2015/4/2 ver.3: 800 nm

%% Used in
% bellaFrogMSFitV02

%% for convenience
rotAgl = cmrPara.rotAgl;
anlThre = cmrPara.anlThre;
timeOS = cmrPara.timeOS;
lmb0 = osPrp.lmb0;

%% main
%img(:,:) = fImgSmooth9(img); % 2D smoothing
% bin + ROI
% [binTime,binWvl,binImg]=fFrogFitBinRoiV01(frgT,frgL,img,binNmb,wvlRoi);
% key pixels
binFrgSz = size(img');   % binned frog image size
timeCntPxl = 0.5*(binFrgSz(1));   % time center in pixle
[~,wvlCntPxl] = min(abs(frgL-0.5*lmb0)); % wavelength center pixel, 2w
% time domain
%timeHlf = frgT(1:0.5*binFrgSz(1));   % time axis (half)
[~,timeOSPxl] = min(abs(frgT-timeOS));   % time+offset (pulse front tilt) [pixel]
% image process
horOS = round(timeOSPxl-timeCntPxl);    % horizontal offset

% figure(11)
% pcolor(img)
% shading interp

[~,symImg] = fImgRotSym(img,[timeOSPxl,wvlCntPxl],horOS,rotAgl,varargin{1}); % rotate, symmetrize, shift
symImg(:,:) = symImg.*(symImg>anlThre);

%% 2D interpolation to omega
orgW = ltow(frgL)- ltow(0.5*lmb0);  % original omega,  - omega0
[orgTM,orgWM] = meshgrid(frgT,orgW); % original time mesh and W mesh
[trgTM,trgWM] = meshgrid(cmrPara.T,cmrPara.W); % target t and w mesh
symLeft = interp2(orgTM,orgWM,symImg,trgTM,trgWM); % 2D interp
mx = max(max(symLeft));

%% test
% figure(10)
% subplot(2,2,1)
% plot(orgW)
% subplot(2,2,2)
% plot(frgT)
% subplot(2,2,3)
% plot(cmrPara.W)
% subplot(2,2,4)
% plot(cmrPara.T)
