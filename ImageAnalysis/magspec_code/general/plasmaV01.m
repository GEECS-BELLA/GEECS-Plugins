c = 3*10^10;    % speed of light, cm/s
nE = [1:1:35];    % electron density, x10^18/c.c.
tE = [10:10:300];%[1:10,20:10:100,200:100:1000];   % temperature of electron plasma [eV]
[x,y] = meshgrid(nE,tE);

bPP = sqrt(8*pi.*x*10^18.*y)/c;

figure(1)
pcolor(nE,tE,bPP)
shading interp
colorbar

%clear all

alp = 0.15*10^4.5/(4*4*10^19);
nI = x;%4*10^19;
z=1;
bNes = alp*z^2.*nI*10^18./(y.^1.5);    %[T]
bNes(end,end)

figure(2)
pcolor(nE,tE,bNes)
%shading interp
colorbar
