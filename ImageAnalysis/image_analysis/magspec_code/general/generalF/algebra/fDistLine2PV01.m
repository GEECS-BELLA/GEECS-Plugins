function d = fDistLine2PV01(inpt)
%% give distance between line and point
%
% lin = fDistLine2PV01(inpt)
%
% inpt: 1x4 array [x0,y0,a,b], y=ax+b for line
%
% d: distance

%
% function distance between line and point, ver.1
%
% Written by Kei Nakamura
% Created on Friday, January 7th 2011.
% Modified on

%% main body

%d = abs(inpt(2)-inpt(3)*inpt(1)-inpt(4))/sqrt(1+inpt(3)^2);
d = abs(inpt(2)-inpt(3).*inpt(1)-inpt(4))/sqrt(1+inpt(3)^2);
