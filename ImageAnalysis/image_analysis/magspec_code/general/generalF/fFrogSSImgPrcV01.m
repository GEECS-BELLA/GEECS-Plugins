function [timeHlf,binWvl,symLeft,mx] = fFrogSSImgPrcV01(frgT,frgL,img,cmrPara,lmb0,varargin)
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

%% Used in
% bellaFrogFitTest
% bellaFrogSSFitV01
%

%% for convenience
binNmb = cmrPara.binNmb;
wvlRoi = cmrPara.wvlRoi;
rotAgl = cmrPara.rotAgl;
anlThre = cmrPara.anlThre;
timeOS = cmrPara.timeOS;

%% main
img(:,:) = fImgSmooth9(img); % 2D smoothing
% bin + ROI
[binTime,binWvl,binImg]=fFrogFitBinRoiV01(frgT,frgL,img,binNmb,wvlRoi);
% key pixels
binFrgSz = size(binImg');   % binne frog image size
timeCntPxl = 0.5*(binFrgSz(1));   % time center in pixle
[~,wvlCntPxl] = min(abs(binWvl-0.5*lmb0)); % wavelength center pixel, 2w
% time domain
timeHlf = binTime(1:0.5*binFrgSz(1));   % time axis (half)
[~,timeOSPxl] = min(abs(binTime-timeOS));   % time+offset (pulse front tilt) [pixel]
% image process
symLeft = fImgRotSym(binImg,[timeOSPxl,wvlCntPxl],round(timeOSPxl-timeCntPxl),rotAgl,varargin{1}); % rotate, symmetrize, shift
symLeft(:,:) = symLeft.*(symLeft>anlThre);

mx = max(max(symLeft));
