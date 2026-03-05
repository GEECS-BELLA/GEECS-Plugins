function [dImg,info] = f12bitPngOpnV04(fPath)
%% read and returns 12bit png image and information
%
% f12bitPngOpnV04 reads 12bit png image at fPath,
% and returns image in double format canceling auto bit depth compensation
% It should work for any-bit image.
%
% [dImg,info] = f12bitPngOpnV04(fPath)
% dImg: output image (double)
% inf: image information
% fPath: file path
%

%% 12bit png file opener ver.4
% Written by Kei Nakamura
%
% 2007/5/9 ver.1: created
% ver.2: return information, bit-depth compensation
% ver.3: in case Significantbits is not applicable.
% 2009/6/3 ver.4: minor modification
%
% used in: ccdLedClbV07.m

%% main
img = imread(fPath,'png');       % img: download image

info = imfinfo(fPath);
if info.SignificantBits>=0
    bitDep = info.SignificantBits;
    bitCmp = 16-bitDep;
else
    bitCmp = 0;
end
cmpFct = 2^bitCmp;
dImg = double(img/cmpFct);      % Double Img: cast to double, back to 12 bit
