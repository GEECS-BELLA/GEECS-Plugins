function [anglC,dAnglC] = fDnnAnglMap(x,y)
%% return angle map for each camera
%
% anglC = fBellaAnglMapAllV01(xC,yC)
%
% anglC: angle cell
% dAnglC: dAngle cell
%
% x,y: structure contains x,y axis info
%   .mm: x (or z) [mm]
%   .dx: dx
%   .incAgl: incident angle to screen [deg]
%   .path: path length [m]
%   .divFX: divergin factor for X plane
%   .divFY: diverging factor for Y plane
%   .accp: acceptance (half angle) [mrad]
%

%% Written by Kei Nakamura
% 2018/2/9 ver.1: created

%% main body

for i=1:4
    % make angleC
    [path,ymm] = meshgrid(x(i).path,y(i).mm);   % path and y mesh
    [divFY,~] = meshgrid(x(i).divFY,y(i).mm);   % divergence mesh
    ymm = ymm./divFY;   % including divergence effect
    path = 1000*atan(0.001*ymm./path); % angle [mrad]
    anglC{i} = path;

    daM = diff(path);   % da matrix
    daM = 0.5*([daM(1,:); daM] + [daM; daM(end,:)]);    % da add 1 extra and average
    dAnglC{i} = daM;
end
