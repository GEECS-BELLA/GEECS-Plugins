%%%%%%%%%% PROGRAM polFit2.m %%%%%%%%%%

% Kei Nakamura
% Created: Monday, January 30th 2006
% This program reads txt data file,
% and does polynominal fitting.

clear all
%close all

fName = 'fct.txt';
a = textread(fName);
a1 = a(1:53,:);
a2 = a(54:10:end,:);
a = [a1;a2];
x=a(:,1)'; y1=a(:,2)'; y2 = a(:,3)';%y = 1000*y;


p2 = polyfit(x,y1,7)
p2b = polyfit(x,y2,7)
%p3 = polyfit(x,y,3)
%p4 = polyfit(x,y,4)
%p5 = polyfit(x,y,6)

xi = linspace(0,820,100);
out1 = polyval(p2,xi);
out2 = polyval(p2b,xi);
%y3 = polyval(p3,xi);
%y4 = polyval(p4,xi);
%y5 = polyval(p5,xi);

figure(1)
subplot(1,2,1)
plot(x,y1,'o',xi,out1)%,xi,y3,'-',xi,y4,'-',xi,y5,'-');
subplot(1,2,2)
plot(x,y2,'o',xi,out2)
p2(1)
p2(2)
p2(3)
p2(4)
p2(5)
p2(6)
p2(7)
p2(8)
p2b(1)
p2b(2)
p2b(3)
p2b(4)
p2b(5)
p2b(6)
p2b(7)
p2b(8)
