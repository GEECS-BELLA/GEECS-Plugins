function [c2c,vgntC] = fDnnLanex(camClb,lanexTtl,lanexData)
%% return count-charge factor and vignette compensation matrix for DNN magspec
%
% [c2c,vgntC] = fDnnLanex(camClb,lanexClbPath)
%
% c2c: counts to charge [fC] coefficients for 1 GeV
% vgntC: vignette compensation cell
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
% lanexClbPath: file path for lanex calibration
%

%% Written by Kei Nakamura
% 2018/2/8 ver.1: created
% 2018/5/1 ver.1b: camera set in camera calibration

%% main body
c2c = zeros(1,4); % allocation for c2c
vgntC = cell(1,4);

for i=1:4
    lanexClb = fLanexClbOutV01(lanexTtl,lanexData,camClb(i).set);   % get lanex calibration out
    [c2c(i),vgntM] = fLanexClbV02(camClb(i),lanexClb,char(camClb(i).screen));
    vgntC{i} = vgntM;
end

c2c = c2c/0.36974;  % varian cameras use F70 BP filter
