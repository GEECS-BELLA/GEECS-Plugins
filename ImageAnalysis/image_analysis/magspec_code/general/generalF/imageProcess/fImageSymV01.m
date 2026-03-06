function [img_out,diffLR] = fImageSymV01(img,varargin)
%% symmetrize image (side way)
% [img_out,diffLR] = fImageSymV01(img)
%
% img_out: symmetrized image in the same format as input image
% diffLR: difference between L and R
%
% img: input image
%

%% Written by Kei Nakamura
% 2016/1/7 ver.1b: created
%

%% main body
[~,sizeX] = size(img);  % image size
if nargin>1
    cntX = varargin{1};
else
    % get centroid
    xPrf = sum(img);    % x profile, integrated in y
    [~,cntX] = fGetRmsV01(1:sizeX,xPrf);  % centroid x
    cntX = round(cntX);
end

% make axis is always overe the center
if cntX<=0.5*sizeX
    img(:,:) = fliplr(img);
    cntX = sizeX - cntX + 1;
end

% get left and right
R1 = img(:,cntX+1:end); % image right 1
[~,sizeX1] = size(R1);  % image size for right1
L1 = img(:,cntX - sizeX1:cntX-1);   % image left 1
L2 = img(:,1:cntX-sizeX1-1);        % image left 2
R2 = fliplr(L2);        % image right 2
img2 = [L2 L1 img(:,cntX) R1 R2];   % combined image
img2 = 0.5*(img2 + fliplr(img2));   % symmetrized image
img_out = img2(:,1:sizeX);      % symmetrized image in original format

if cntX<0.5*sizeX
  img_out(:,:) = fliplr(img_out);
end

% left-right difference
LR = abs([L2 L1] - fliplr([R1 R2]));
diffLR = sum(sum(LR));
