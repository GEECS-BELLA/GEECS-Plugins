function [d2,d3,d4,slp2,slp3,slp4] = fCpaDispersionV01(L,N,lmb0,alp,sgn)
%% return dispersions for CPA system
% [d2 d3 d4] = fCpaDispersionV01(L,N,lmb0,alp,sgn)
%
% d2 - d4: 2, 3, and 4th order dispersion
%
% L: optical path length between gratings [mm]
% N: groove density [nmb/mm]
% lmb0: central wavelength [nm]
% alp: incident angle [deg]
% sgn: 1 for stretcher, -1 for compressor (negative chirp)
%

%% Written by Kei Nakamura
% 2013/8/1 ver.1: created
% 2013/10/8 ver.1.1: add slopes
% 2014/1/3 ver,1.2: add kei equation
% 2014/2/26 vrer 1.3: corrected factor 2 for slp4

%% Used in
% bellaSsaSimV03
% bellaFrogPhaseMSV02
% bellaCompSlopeV02

%% constants

c = 0.000299792458; % speed of light [mm/fs]
lmb0 = lmb0*1e-6;    % convert from nm to mm
alp = alp*pi/180;   % convert to radian
bta = asin(N*lmb0 - sin(alp));    % difflection angle [rad]

%% check with labview program
% compD = L*cos(bta);
% K = compD*N*N/(c^2*pi*2);
% KF = 2*K*(lmb0/cos(bta))^3;
% G = (1+sin(alp)*sin(bta))/cos(bta)^2;
% H = (cos(alp)/cos(bta))^2;
% J = lmb0/(2*pi*c);
% p3 = KF*J*3*G;
% p4 = KF*J*J*3*(H-5*G*G);

%% calcluation
slp2 = sgn*N*N*lmb0^3/(c^2*pi*cos(bta)^2); % slope for GVD mm^2 / mm
slp3 = -3*slp2*lmb0*(1 + sin(alp)*sin(bta))/(2*c*pi*cos(bta)^2); % slope for third ord disp mm^3/mm
% csaba equation
slp4 = -3*slp2*lmb0^2*(cos(alp)^2*cos(bta)^2-5*(1+sin(alp)*sin(bta))^2)/(4*pi*pi*c*c*cos(bta)^4);


d2 = slp2*L; % GVD mm^2
d3 = slp3*L; % third ord disp mm^3
d4 = slp4*L;
