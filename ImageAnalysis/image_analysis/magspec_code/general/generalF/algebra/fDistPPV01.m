function d = fDistPPV01(inpt)
%% give line for given 2 points
% d = fDistPPV01(inpt)
%
% inpt: 1x4 array [x1,y1,x2,y2]
%
% d: distance between 2 points

%
% distance between 2 points, ver.1
%
% Written by Kei Nakamura
% Created on Firday, January 7th 2011.
% Modified on

%% main body

d = sqrt((inpt(3)-inpt(1))^2 + (inpt(4)-inpt(2))^2);
