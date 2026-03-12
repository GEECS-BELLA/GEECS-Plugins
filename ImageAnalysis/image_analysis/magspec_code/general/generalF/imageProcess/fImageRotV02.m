function img_out = fImageRotV02(img_in,rotDeg,varargin)
%% img_out = fImageRotV02(img_in,rotDeg,pvt)
% rotate image around pivot pixel
%
% img_in:      input image
% rotDeg:      rotation angle in degree
% pvt:          pivot pixel (x,y)
%
% img_out:     output image

% Function image rotate
% Written by Kei Nakamura
% ver.1: 2011/10/28
% Created
% ver.2: 2012/3/8
% add pivot


%% main body

% meshgrid for original image, place 0 at the pivot
[szY,szX] = size(img_in);   % size
[xLoc,yLoc] = meshgrid(1:szX,1:szY);
if nargin==3
    pvt = varargin{1};
else
    pvt(1) = round(0.5*szX);
    pvt(2) = round(0.5*szY);
end
xLoc(:,:) = xLoc - pvt(1);
yLoc(:,:) = yLoc - pvt(2);

% make padded image
padImg = zeros(2*szY,2*szX);
initX = round(0.5*szX); initY = round(0.5*szY); % initial x and y
padImg(initY:initY+szY-1,initX:initX+szX-1) = img_in;

% figure(4)
% subplot(2,2,1)
% pcolor(padImg)
% shading interp
% colorbar

% meshgrid for padded image, place 0 at the pivot
[xLocP,yLocP] = meshgrid(1:2*szX,1:2*szY);
xLocP(:,:) = xLocP - (pvt(1)+initX-1);
yLocP(:,:) = yLocP - (pvt(2)+initY-1);

% x,y location for target image
rotRad = rotDeg*pi/180; % rotation angle in radian
xTrgt = xLoc*cos(rotRad)-yLoc*sin(rotRad);
yTrgt = xLoc*sin(rotRad)+yLoc*cos(rotRad);

% 2d interpolation
img_out = interp2(xLocP,yLocP,padImg,xTrgt,yTrgt);

% subplot(2,2,2)
% pcolor(img_out)
% shading interp
% colorbar
