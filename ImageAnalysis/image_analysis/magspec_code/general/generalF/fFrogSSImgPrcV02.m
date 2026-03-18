function [symLeft,mx] = fFrogSSImgPrcV02(frgT,frgL,img,cmrPara,lmb0,varargin)
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
% varagin {1}: figure handle for figure on
%

%% Written by Kei Nakamura
% 2014/8/1 ver.1: created
% 2014/11/18 ver.2: for 3x3 binned image

%% Used in
% bellaFrogMSFitV02

%% for convenience
rotAgl = cmrPara.rotAgl;
anlThre = cmrPara.anlThre;
timeOS = cmrPara.timeOS;

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
[~,symImg] = fImgRotSym(img,[timeOSPxl,wvlCntPxl],round(timeOSPxl-timeCntPxl),rotAgl,varargin{1}); % rotate, symmetrize, shift
symImg(:,:) = symImg.*(symImg>anlThre);

%% 2D interpolation to omega
orgW = ltow(frgL);
[orgTM,orgWM] = meshgrid(frgT,orgW); % original time mesh and W mesh
[trgTM,trgWM] = meshgrid(cmrPara.T,cmrPara.W2); % target t and w mesh
symLeft = interp2(orgTM,orgWM,symImg,trgTM,trgWM); % 2D interp
mx = max(max(symLeft));
