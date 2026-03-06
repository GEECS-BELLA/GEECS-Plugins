function d = fDistLine2LineV01(x,y,linAB)
%% give distance between line and point
%
% lin = fDistLine2PV01(inpt)
%
% inpt: x array
%       y array
%       linAB: ab for line
%
% d: distance
%
%
% function distance between line and point, ver.1
%
% Written by Kei Nakamura
% 2019/3/10, ver.1
% Modified on

%% main body

d = abs(y - linAB(1)*x - linAB(2))./sqrt(1+linAB(1)^2);

%d = abs(inpt(2)-inpt(3)*inpt(1)-inpt(4))/sqrt(1+inpt(3)^2);
