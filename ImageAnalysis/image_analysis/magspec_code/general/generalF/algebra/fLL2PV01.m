function p = fLL2PV01(a1,b1,a2,b2)
%% give point where 2 lines are crossed
% p = fLL2PV01(a1,b1,a2,b2)
% a1-b2: a and b for lines (y=ax+b)
% p: point [x,y]

% p = fLL2PV01(1,1,2,-2) gives point where y=x+1 and y=2x-2 cross
%
% 2 lines to point, ver.1
%
% Written by Kei Nakamura
% Created on Thursday, June 25th 2009.
% Modified on

%% main body

x = (b2-b1)./(a1-a2);
y = a1.*x+b1;
p = [x,y];
