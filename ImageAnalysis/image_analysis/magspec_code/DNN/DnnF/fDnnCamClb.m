function camClb = fDnnCamClb(camTtl,camData)
%% return camera calibration for DNN magspec
%
% camClb = fDnnCamClb(camTtl,camData)
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
%   .rot: rotation [deg]
%
% camTtl,camData: title and data for bella magspec camara calibration
%

%% Written by Kei Nakamura
% 2018/2/8 ver.1: created
% 2018/5/1 ver.1b: set and screen
% 2018/11/19 ver.1c: add name

%% main body

% allocation
[szy,~] = size(camData);
camClb(szy).fov = 0;

for i=1:szy
    camClb(i).fov = str2double(camData(i,fLogClmnFindV01(camTtl,'FOV [mm]'))); % fov column
    camClb(i).yOffset = str2double(camData(i,fLogClmnFindV01(camTtl,'ROI Y offset'))); %
    camClb(i).height = str2double(camData(i,fLogClmnFindV01(camTtl,'ROI height'))); %
    camClb(i).xOffset = str2double(camData(i,fLogClmnFindV01(camTtl,'ROI X offset'))); %
    camClb(i).width = str2double(camData(i,fLogClmnFindV01(camTtl,'ROI width'))); %
    camClb(i).leftPos = str2double(camData(i,fLogClmnFindV01(camTtl,'Left edge [mm]'))); %
    camClb(i).yCntr = str2double(camData(i,fLogClmnFindV01(camTtl,'Y center pixel'))); %
    camClb(i).ySt = str2double(camData(i,fLogClmnFindV01(camTtl,'Y Start'))); %
    camClb(i).xSt = str2double(camData(i,fLogClmnFindV01(camTtl,'X Start'))); %
    camClb(i).yEd = str2double(camData(i,fLogClmnFindV01(camTtl,'Y End'))); %
    camClb(i).xEd = str2double(camData(i,fLogClmnFindV01(camTtl,'X End'))); %
    camClb(i).rot = str2double(camData(i,fLogClmnFindV01(camTtl,'rot [deg]'))); %
    camClb(i).set = str2double(camData(i,fLogClmnFindV01(camTtl,'set',1))); %
    camClb(i).screen = camData(i,fLogClmnFindV01(camTtl,'screen')); %
    camClb(i).name = camData(i,fLogClmnFindV01(camTtl,'name')); %
    camClb(i).sensitivity = str2double(camData(i,fLogClmnFindV01(camTtl,'sensitivity'))); %
end
