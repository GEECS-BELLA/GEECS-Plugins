function dsprsn = fCpaDispersionV03(D,N,lmb0,alp,sgn,dspOrd)
%% return dispersions for CPA system
% dsprsn = fCpaDispersionV02(L,N,lmb0,alp,sgn,ord)
%
% dsprsn: array with polynomial coefficients, (1) for 6th, (7) for 0th.
%
% D: distance between gratings [mm]
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
% 2015/9/24 ver.3: 12th order

%% Used in

%% constants
c = 0.000299792458; % speed of light [mm/fs]
lmb0 = lmb0*1e-6;    % convert from nm to mm
alp = alp*pi/180;   % convert to radian
bta = asin(N*lmb0 - sin(alp));    % difflection angle [rad]
%disp([bta, bta*180/pi])

%% to make it simple
A = D*N^2/(pi*c^2);
Y = lmb0/(2*pi*c);
F = (lmb0/cos(bta))^3;
G = (1+sin(alp)*sin(bta))/cos(bta)^2;
H = (cos(alp)/cos(bta))^2;

%% dispersions
dsprsn = zeros(numel(D),13);
dsprsn(:,11) = -A*F;    % 2nd
dsprsn(:,10) = 3*Y*A*F*G;   % 3rd
dsprsn(:,9) = 3*Y^2*A*F*(H - 5*G^2);   % 4th
dsprsn(:,8) = 15*Y^3*A*F*G*(-3*H + 7*G^2);  % 5th
dsprsn(:,7) = 45*Y^4*A*F*(-H^2 + 14*G^2*H -21*G^4);  % 6th
dsprsn(:,6) = 315*Y^5*A*F*G*(5*H^2 - 30*G^2*H +33*G^4);  % 7th
dsprsn(:,5) = 315*Y^6*A*F*(5*H^3 - 135*G^2*H^2 + 495*G^4*H -429*G^6);  % 8th
dsprsn(:,4) = 2835*Y^7*A*F*G*(-35*H^3 +385*G^2*H^2 -1001*G^4*H +715*G^6);  % 9th
dsprsn(:,3) = 14175*Y^8*A*F*(-7*H^4 +308*G^2*H^3 -2002*G^4*H^2 +4004*G^6*H -2431*G^8);  % 10th
dsprsn(:,2) = 155925*Y^9*A*F*G*(63*H^4 -1092*G^2*H^3 +4914*G^4*H^2 -7956*G^6*H +4199*G^8);  % 11th
dsprsn(:,1) = 155925*Y^10*A*F*(63*H^5 -4095*G^2*H^4 +40950*G^4*H^3 -139230*G^6*H^2 +188955*G^8*H -88179*G^10);  % 12th

%% order
dsprsn(:,1:12-dspOrd) = 0;    % put 0 for the order not to be considered

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
