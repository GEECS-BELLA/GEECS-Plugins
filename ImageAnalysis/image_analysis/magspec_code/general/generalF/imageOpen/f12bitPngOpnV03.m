%%%%%%%%%% PROGRAM f12bitPngOpnV03.m %%%%%%%%%%
% 12bit png file opener ver.3
% Kei Nakamura
% Created on Wednesday, May 9th 2007.
% Modified on Monday, May 14th 2007.

% This program open 12bit png image file,
% and returns it as double format.

% img: output image
% fNam: file name

% ver.2: return information, bit-depth compensation
% ver.3: in case Significantbits is not applicable.

function [dImg inf] = f12bitPngOpnV03(fNam)  %function header

img = imread(fNam,'png');       % img: download image

inf = imfinfo(fNam);
if inf.SignificantBits>=0;
    bitDep = inf.SignificantBits;
    bitCmp = 16-bitDep;
else
    bitCmp = 0;
end
cmpFct = 2^bitCmp;
dImg = double(img/cmpFct);      % Double Img: cast to double, back to 12 bit
