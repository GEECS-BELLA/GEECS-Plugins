% n20 = 2.5;
% N1 = 2.3;
% N2 = 1;
% l0 = 266;
% l = 804;
% l1 = 46.6;
% l2 = 1086.3;
%
% exp1 = N1*exp(-(l-l0)/l1)
% exp2 = N2*exp(-(l-l0)/l2)
% n2 = n20 + exp1 +exp2

n2 = 1.45332;
clckAnglD = [0:1:180];
clckAngl = clckAnglD * pi / 180;    % clocking angle [rad]

wdgAnglHD = 2*cos(clckAngl);    % wedge angle horizontal [deg]
wdgAnglVD = 2*sin(clckAngl);    % wedge angle vertical [deg]
wdgAnglH = wdgAnglHD*pi/180;    % wedge angle horizontal [rad]
wdgAnglV = wdgAnglVD*pi/180;    % wedge angle vertical [rad]
%wdgAngl = 0:0.1:5;    % wedge angle [deg]

incAnglHD = 5;
incAnglH = incAnglHD*pi/180; % incident angle [rad]
intAnglH1 = asin(sin(incAnglH)/n2);   % internal angle 1 [rad]
intAngl1HD = intAnglH1*180/pi;
intAngl2HD = intAngl1HD + 2*wdgAnglHD;
intAngl2H = intAngl2HD*pi/180;
outAnglH = asin(n2*sin(intAngl2H));
outAnglHD = outAnglH*180/pi;
sepAnglHD = outAnglHD - incAnglHD;

incAnglVD = 0;
incAnglV = incAnglVD*pi/180; % incident angle [rad]
intAnglV1 = asin(sin(incAnglV)/n2);   % internal angle 1 [rad]
intAngl1VD = intAnglV1*180/pi;
intAngl2VD = intAngl1VD + 2*wdgAnglVD;
intAngl2V = intAngl2VD*pi/180;
outAnglV = asin(n2*sin(intAngl2V));
outAnglVD = outAnglV*180/pi;
sepAnglVD = outAnglVD - incAnglVD;

figure(1)
plot(clckAnglD,sepAnglHD)
hold on
plot(clckAnglD,sepAnglVD,'r-')
plot(clckAnglD,sqrt(sepAnglHD.^2+sepAnglVD.^2),'g-')
hold off

figure(2)
plot(clckAnglD,sqrt(sepAnglHD.^2+sepAnglVD.^2),'g-')

%p = polyfit(outAnglHD-5,wdgAngl,1);
%crAngl = polyval(p,6.5532);
