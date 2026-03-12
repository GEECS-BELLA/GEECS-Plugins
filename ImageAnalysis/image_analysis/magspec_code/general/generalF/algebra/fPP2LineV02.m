function lin = fPP2LineV02(inpt)
%% give line for given 2 points
% lin = fPP2LineV01(inpt)
%
% inpt: 1x4 array [x1,y1,x2,y2]
%
% lin: line [a,b] for y=ax+b

% lin = fPP2LineV01(1,1,3,3) gives line go through (1,1) and (3,3).
%
% 2 points to line, ver.1
%
% Written by Kei Nakamura
% Created on Thursday, June 25th 2009.
% Modified on

%% main body

a = (inpt(4)-inpt(2))/(inpt(3)-inpt(1));
b = inpt(2) - a*inpt(1);
lin = [a,b];
