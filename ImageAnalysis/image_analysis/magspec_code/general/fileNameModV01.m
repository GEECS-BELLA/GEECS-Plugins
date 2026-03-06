%% This program change file name
% PROGRAM bellaFileNameModV01.m
% BELLA File Name change program

%% Written by Kei Nakamura
% 2013/3/28 ver.1: created

%% initializations
clear all;

%% input parameters


%% directories
dir1 = '/Users/kee/Documents/data/Staging/13_0418/scans/Scan045/MagCam1';
dir2 = '/Users/kee/Documents/data/Staging/13_0418/scans/Scan045/MagCam1b';
mkdir(dir2);

list = dir([dir1,'/S*']);
%%


for ii = 1:numel(list)
    srcName = [dir1,'/',list(ii).name];
    dstName = list(ii).name;
    dstName = [dstName(1:end-3),'.',dstName(end-2:end)];
    dstName = [dir2,'/',dstName];
    movefile(srcName,dstName)
end

%%
[rDir,clbDir] = fBellaRootDirV01(day,mode);  % mother directory
scnDir = [rDir,'/scans']; % scan directory
scanNS = fGet3NmbStringV01(scanN);  % 3 digit scan number string
scnNDir = [scnDir,'/scan',scanNS];            % scan# directory
imgPathC = fBellaImgDir(scnNDir,scanNS); % cell of image path

%%
dir1 = imgPathC{3};
dirS = [dir1(1:68)];
dirD = [dir1(1:67),'temp/'];
list = dir([imgPathC{3},'*.png']);


%

                                 % number of data out for angle txt
