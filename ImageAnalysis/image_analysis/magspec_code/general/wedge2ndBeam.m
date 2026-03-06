%% To estimate 2nd beam from wedge
%
% PROGRAM wedge2ndBeam.m
%

%% Revisions
% Written by Kei Nakamura
% 2017/10/13 Ver.1: created

%% input
n0 = 1;     % refractive index for input/output side
wvl = 800;   % wavelength
angInD = 40; % angle in [deg]
wdgAngD = 0.5;   % wedge angle [deg]
thck = 0.0254*3;  % thickness of the optis [m]
prpL = 11.34;      % propagation length [m]
mat = 'fused silica';
%mat = 'BK7';

%% constant
angInR = angInD*pi/180;
wdgAngR = wdgAngD*pi/180;
n1 = refindex(wvl,mat);  % refractive index for substrate

%% calculation
angS1R = asin(n0*sin(angInR)/n1);   % angle s1 [rad]
angS1D = angS1R * 180/pi;           % angle s1 [deg]
angS3R = angS1R + 2*wdgAngR;        % angle s3 [rad]
angO2R = asin(n1*sin(angS3R)/n0);   % output angle 2 [rad]
angO2D = angO2R*180/pi;             % output angle 2 [deg]
dAngD = angO2D - angInD;            % delta-angle [deg]
dAngR = dAngD*pi/180;               % delta-angle [rad]
dSrfc = 1000*thck*(tan(angS1R)+tan(angS1R+2*wdgAngR));   % delta at surface [mm]
dSpt = 1000*prpL*tan(dAngR);             % delta-spot [mm]
sExtr = 1000*n1*thck*(1/cos(angS1R)+1/cos(angS1R+2*wdgAngR)); % extra optical path[mm]

%% display
disp(['delta angle: ' num2str(dAngD) ' [deg]'])
disp(['displacement at surface: ' num2str(dSrfc) ' [mm]'])
disp(['displacement at focus: ' num2str(dSpt) ' [mm]'])
disp(['extra optical path length: ' num2str(sExtr) ' [mm]'])
