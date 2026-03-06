function [x,y] = fDrawCircle(cntr,rds)
%% give x y for a circle
%
% Written by Kei Nakamura
% 2019/1/4 Ver.1: created

%% main body

aaa = linspace(0,2*pi,100);
x = rds*cos(aaa)+cntr(1);
y = rds*sin(aaa)+cntr(2);
