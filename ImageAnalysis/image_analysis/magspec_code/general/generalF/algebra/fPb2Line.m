function line = fPb2Line(p,b)
%% give line for given point and b (y=ax+b)
% line = fPb2Line(p,b)
%
% p: [x,y]
% b: from y = ax+b
%
% line: line [a,b] for y=ax+b
%
% Written by Kei Nakamura
% 2014/4/5 ver.1: Created

%% main body

a = (p(2) - b)/p(1);
line = [a,b];
