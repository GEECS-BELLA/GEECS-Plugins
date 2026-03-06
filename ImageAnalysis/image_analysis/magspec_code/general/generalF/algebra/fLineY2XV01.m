function x = fLineY2XV01(ln,y)
%% give y for given x and line
% x = fLineY2XV01(ln,y)
% ln: line, [a,b] for y=ax+b
% x: x coordinate
% y: y coordinate
% x = fLineY2XV01([1,1],2) gives x=y-1 for y=2.
%
% line y to x, ver.1
%
% Written by Kei Nakamura
% Created on Thursday, June 25th 2009.
% Modified on

%% main body

x = (y - ln(2))/ln(1);  % x = (y-b)/a
