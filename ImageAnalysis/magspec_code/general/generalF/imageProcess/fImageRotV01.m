function img_out = fImageRotV01(img_in,rotDeg)
%% image rotation
%
% img_out = fImageRotV01(img_in,rotDeg)
%
% img_in:      input image
% rotDeg:      rotation angle in degree
%
% img_out:     output image

% Kei Nakamura
% Created on Friday, October 28th 2011.

%% main body

% meshgrid for original image, place 0 at the center of the image
[szY,szX] = size(img_in);   % size
[xLoc,yLoc] = meshgrid(1:szX,1:szY);
xLoc(:,:) = xLoc - round(0.5*szX);
yLoc(:,:) = yLoc - round(0.5*szY);

% make padded image
padImg = zeros(2*szY,2*szX);
initX = round(0.5*szX); initY = round(0.5*szY); % initial x and y
padImg(initY:initY+szY-1,initX:initX+szX-1) = img_in;

% meshgrid for padded image, place 0 at the center of the image
[xLocP,yLocP] = meshgrid(1:2*szX,1:2*szY);
xLocP(:,:) = xLocP - szX;
yLocP(:,:) = yLocP - szY;

% x,y location for target image
rotRad = rotDeg*pi/180; % rotation angle in radian
xTrgt = xLoc*cos(rotRad)-yLoc*sin(rotRad);
yTrgt = xLoc*sin(rotRad)+yLoc*cos(rotRad);

% 2d interpolation
img_out = interp2(xLocP,yLocP,padImg,xTrgt,yTrgt);
