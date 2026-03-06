function [c2c,vgntM] = fLanexClbV01(camInfo,lanexClb,screen)
%% return count-charge factor and darkening (vignetting) compensation matrix for LANEX
%
% [alsR,vgntM] = fLanexClbV01(setting,fov,roiInfo)
%
% c2c: counts to charge [fC] coefficients for 1 GeV
% vgntM: vignetting compensation matrix
%
% camInfo: array of camera information.
%          (1)y-start (2)y-end (3)x-start (4)x end (5)fov
% lanexClb: lanex calibration data array,
%       [1]    'FOV slope'
%       [2]    'FOV offset'
%       [3]    'vignette 4'
%       [4]    'vignette 2'
%       [5]    'vignette 0'
%       [6]    'sensitivity 2'
%       [7]    'sensitivity 1'
%       [8]    'sensitivity 0'
%       [9]    'full width' of the camera pixel
%       [10]    'full height' of the camera pixel
% screen: 'front' (lanex fast front, thin) or 'back' (LFB, thick)

%% Written by Kei Nakamura
% 2013/3/18 ver.1:, created

%% for c2c

% screen factor
if numel(screen)==4 % 'back' is 4 letters
    screenF = 1;
else screenF = 1.98;
end

% camera - screen distance Z
z = (camInfo(5) - lanexClb(2))/lanexClb(1);

% als ratio (less#, more sensitive)
alsR = lanexClb(6)*z^2 + lanexClb(7)*z + lanexClb(8);

% counts to charge [fC] factor for 1 GeV [fC/count]
c2c = screenF*alsR/146;

%% for vignette compensation matrix

[aaa,bbb] = meshgrid(1:lanexClb(9),1:lanexClb(10));    % original image size
aaa(:,:) = aaa - lanexClb(9)/2+0.5;
bbb(:,:) = bbb - lanexClb(10)/2+0.5; % putting center - 0
aaa(:,:) = sqrt(aaa.^2 + bbb.^2);            % pixel r map

vgntM = aaa(camInfo(1):camInfo(2),camInfo(3):camInfo(4));     % roi
vgntM(:,:) = lanexClb(3)*vgntM.^4 + lanexClb(4)*vgntM.^2 + lanexClb(5);   % compensation
vgntM(:,:) = 1./vgntM;
