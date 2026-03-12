%% input
clear all; close all;
beamPrm = [0.01 0 0.01 0 0 0]; % beam parameter
nmbPrt = 1e6;   % # of particles

%% main
% make beam
beam0 = fRandBeam(beamPrm,1e6);
% plot
plotPrt(1,beam0)

%% layer 1 calculation
theta(1) = 0.001;   % scattering angle of layer 1
x(1) = 0.01;        % x (propagation length) of layer 1
beam1 = fSctBeam(beam0,theta(1),x(1));
avgXangl(1) = mean(abs(beam1(:,2)));
avgYangl(1) = mean(abs(beam1(:,4)));
avgAngl(1) = (avgXangl(1) + avgYangl(1))/2;
engFct(1) = 1/cos(avgAngl(1));
disp(engFct(1))
% plot
plotPrt(2,beam1)

%% layer 2 calculation
theta(2) = 0.001;   % scattering angle of layer 1
x(2) = 0.01;        % x (propagation length) of layer 1
beam2 = fSctBeam(beam1,theta(2),x(2));
avgXangl(2) = mean(abs(beam2(:,2)));
avgYangl(2) = mean(abs(beam2(:,4)));
avgAngl(2) = (avgXangl(2) + avgYangl(2))/2;
engFct(2) = 1/cos(avgAngl(2));
disp(engFct(2))
% plot
plotPrt(3,beam2)


%% functions

function beam0 = fRandNBeam(beamPrm,numPrt)
% generate Gaussian distribution beam
%
% beamPrm: beam paramter
% [sigma_x, sigma_x', sigma_y, sigma_y', sigma_l, sigma_dp]

x0 = beamPrm(1)*randn(numPrt,1);
xp0 = beamPrm(2)*randn(numPrt,1);
y0 = beamPrm(3)*randn(numPrt,1);
yp0 = beamPrm(4)*randn(numPrt,1);
l0 = beamPrm(5)*randn(numPrt,1);
p0 = beamPrm(6)*randn(numPrt,1);
beam0 = [x0 xp0 y0 yp0 l0 p0]; % matrix for beam,
end

function beam0 = fRandBeam(beamPrm,numPrt)
% generate flat-top distribution beam
%
% beamPrm: beam parameter [x size(r), x', y, y', l, dp]
%

x0 = beamPrm(1)*rand(numPrt,1);
xp0 = beamPrm(2)*rand(numPrt,1);
y0 = beamPrm(3)*rand(numPrt,1);
yp0 = beamPrm(4)*rand(numPrt,1);
l0 = beamPrm(5)*rand(numPrt,1);
p0 = beamPrm(6)*rand(numPrt,1);
beam0 = [x0 xp0 y0 yp0 l0 p0]; % matrix for beam,
end

function beam1 = fSctBeam(beam0,theta,x)
[numPrt,~] = size(beam0);
%x
randN1 = randn(numPrt,1);
randN2 = randn(numPrt,1);
beam1(:,1) = beam0(:,1) + x*theta/sqrt(12)*randN1 + x*theta/2*randN2;
beam1(:,2) = beam0(:,2) + theta*randN2;
%y
randN1 = randn(numPrt,1);
randN2 = randn(numPrt,1);
beam1(:,3) = beam0(:,3) + x*theta/sqrt(12)*randN1 + x*theta/2*randN2;
beam1(:,4) = beam0(:,4) + theta*randN2;
end

function plotPrt(figN,beam)
figure(figN)
subplot(1,2,1)
histogram2(1e3*beam(:,1),1e3*beam(:,3),'DisplayStyle','tile');
xlabel('x_0 [mm]')
ylabel('y_0 [mm]')
title('x-y distribution')
axis equal
subplot(1,2,2)
histogram2(1e3*beam(:,2),1e3*beam(:,4),'DisplayStyle','tile');
xlabel('xp [mrad]')
ylabel('yp [mrad]')
title('x-y angle distribution')
axis equal
end
