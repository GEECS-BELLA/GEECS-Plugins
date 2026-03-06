function dsprsn = fCpaDispersionV02(D,N,lmb0,alp,sgn,dspOrd)
%% return dispersions for CPA system
% dsprsn = fCpaDispersionV02(L,N,lmb0,alp,sgn,ord)
%
% dsprsn: array with polynomial coefficients, (1) for 6th, (7) for 0th.
%
% D: distabce between gratings [mm]
% N: groove density [nmb/mm]
% lmb0: central wavelength [nm]
% alp: incident angle [deg]
% sgn: 1 for stretcher, -1 for compressor (negative chirp)
% dspOrd: order of dispersion to be considered.

%% Written by Kei Nakamura
% 2013/8/1 ver.1: created
% 2013/10/8 ver.1.1: add slopes
% 2014/1/3 ver,1.2: add kei equation
% 2014/2/26 ver 1.3: corrected factor 2 for slp4
% 2014/3/4 ver.2: uniform output, 6th order

%% Used in

%% constants
c = 0.000299792458; % speed of light [mm/fs]
lmb0 = lmb0*1e-6;    % convert from nm to mm
alp = alp*pi/180;   % convert to radian
bta = asin(N*lmb0 - sin(alp));    % difflection angle [rad]

%% to make it simple
X = D*N*N/(c^2*pi);
Y = lmb0/(2*pi*c);
F = (lmb0/cos(bta))^3;
FX = F*X;
G = (1+sin(alp)*sin(bta))/cos(bta)^2;
H = (cos(alp)/cos(bta))^2;

%% dispersions
dsprsn = zeros(numel(D),7);
dsprsn(:,1) = 45*FX*Y^4*(14*G^2*H - 21*G^4 - H^2);  % 6th
dsprsn(:,2) = 15*FX*Y^3*(7*G^3 - 3*G*H);  % 5th
dsprsn(:,3) = 3*FX*Y^2*(H - 5*G^2);   % 4th
dsprsn(:,4) = 3*FX*Y*G;   % 3rd
dsprsn(:,5) = -FX;    % 2nd

%% order
dsprsn(:,1:6-dspOrd) = 0;    % put 0 for the order not to be considered

%% sign
dsprsn(:,:) = -1*sgn*dsprsn;  % flip sign for stretcher

%% previous version
% slp2 = sgn*N*N*lmb0^3/(c^2*pi*cos(bta)^2); % slope for GVD mm^2 / mm
% slp3 = -3*slp2*lmb0*(1 + sin(alp)*sin(bta))/(2*c*pi*cos(bta)^2); % slope for third ord disp mm^3/mm
% slp4 = -3*slp2*lmb0^2*(cos(alp)^2*cos(bta)^2-5*(1+sin(alp)*sin(bta))^2)/(4*pi*pi*c*c*cos(bta)^4);
%
% d2 = slp2*L; % GVD mm^2
% d3 = slp3*L; % third ord disp mm^3
% d4 = slp4*L;
