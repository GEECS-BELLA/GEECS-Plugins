function line = fPa2Line(p,a)
%% give line for given point and a (slope)
% lin = fPP2LineV01(inpt)
%
% p: [x,y]
% a: slope for the line
%
% lin2: line [a,b] for y=ax+b
%
% Written by Kei Nakamura
% 2014/4/5 ver.1: Created

%% main body

b = p(2) - a*p(1);
line = [a,b];
