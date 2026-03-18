function [c2c,vgntM] = fLanexClbV03(camClb,lanexClb,screen,roi)
%% return count-charge factor and darkening (vignetting) compensation matrix for LANEX
%
% [c2c,vgntM] = fLanexClbV02(camClb,lanexClb,screen)
%
% c2c: counts to charge [fC] coefficients for 1 GeV
% vgntM: vignetting compensation matrix
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
% lanexClb: structure containing lanex calibration info
%   .fovSlp: fov slope
%   .fovOffst: FOV offset
%   .vgnt4: vignette 4th order coeff.
%   .vgnt2: vignette 2nd order coeff.
%   .vgnt0: vignette 0th order coeff.
%   .sense2: sensitivity 2nd order coeff.
%   .sense1: sensitivity 1st order coeff.
%   .sense0: sensitivity oth order coeff.
%   .width: full width of the camera pixel
%   .height: full height of the camera pixe
%
% screen: 'front' (lanex fast front, thin) or 'back' (LFB, thick)
% roi: analysis roi on/off, put 0 to stop roi

%% Written by Kei Nakamura
% 2013/3/18 ver.1: created
% 2013/3/20 ver.2: structure
% 2013/4/23 ver.3: make analysis roi optional

%% for c2c

% analysis roi, put anything in input to stop it
if nargin==4
    anlROI = roi;
else anlROI = 1;
end

% screen factor
if numel(screen)==4 % 'back' is 4 letters
    screenF = 1;
else screenF = 1.98;
end

% camera - screen distance Z
z = (camClb.fov - lanexClb.fovOffst)/lanexClb.fovSlp;

% als ratio (less#, more sensitive)
alsR = lanexClb.sense2*z^2 + lanexClb.sense1*z + lanexClb.sense0;

% counts to charge [fC] factor for 1 GeV [fC/count]
c2c = screenF*alsR/146;

%% for vignette compensation matrix
ySt = camClb.yOffset + 1;               % roi y start
yEd = camClb.yOffset + camClb.height;    % roi y end
xSt = camClb.xOffset + 1;               % roi x start
xEd = camClb.xOffset + camClb.width;     % roi x end

[aaa,bbb] = meshgrid(1:lanexClb.width,1:lanexClb.height);    % original image size
aaa(:,:) = aaa - lanexClb.width/2+0.5;
bbb(:,:) = bbb - lanexClb.height/2+0.5; % putting center - 0
aaa(:,:) = sqrt(aaa.^2 + bbb.^2);            % pixel r map

vgntM = aaa(ySt:yEd,xSt:xEd);     % roi (saving)
if anlROI==1
vgntM = vgntM(camClb.ySt:camClb.yEd,camClb.xSt:camClb.xEd);   % roi (analysis)
end
vgntM(:,:) = lanexClb.vgnt4*vgntM.^4 + lanexClb.vgnt2*vgntM.^2 + lanexClb.vgnt0;   % compensation
vgntM(:,:) = 1./vgntM;
