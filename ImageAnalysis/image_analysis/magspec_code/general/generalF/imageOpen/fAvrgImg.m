function img = fAvrgImg(scnNDir,diagNam)
%% read all images in a folder, return averaged one
%
% It works for GEECS format.
%
% img = fAvrgImg(scanNDir,diagNam)
%
% scnNDir: path for scan number dir
% diagNam: diagnostic name
%
% img: averaged image

%% Written by Kei Nakamura
% 2018/4/6 ver.1: created
%

%% main
camDir = [scnNDir,filesep,diagNam]; % image folder
imgList = dir([camDir,filesep,'*',diagNam,'*.png']);    % image list
nmbImg = numel(imgList);    % number of images
imgPath = [camDir,filesep,imgList(1).name];
img = f12bitPngOpnV04(imgPath);

for jj = 2:nmbImg
    %imgPath = [imgList(jj).folder,filesep,imgList(jj).name];
    imgPath = [camDir,filesep,imgList(jj).name];
    img(:,:) = img + f12bitPngOpnV04(imgPath);
end
img(:,:) = img/numel(imgList);
