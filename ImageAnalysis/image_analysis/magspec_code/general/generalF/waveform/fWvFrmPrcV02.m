%%%%%%%%%% Function fWvFrmPrcV02.m %%%%%%%%%%
% Wave Form Process ver.2
% Kei Nakamura
% Created on Tuesday, May 8th 2007.
% Modified on Tuesday, May 15th 2007.

% This function reads text file,
% which recorded waveform of ICT signal at ALS experiments,
% and return the charge.

% chrg: charge [pC]
% sn: S/N ratio
% fNam: file name to open
% tim: time for background evaluation (tim(1)-tim(2)) and
%       time for signal evaluation (tim(3)-time(4)).

% ver.2: return S/N ratio as well

function [chrg sn] = fWvFrmPrcV02(fNam, tim)  % function header

wf = textread(fNam);                           % download waveform
timStp = wf(2,1)-wf(1,1);                       % time step

% find index for times
[delete timI(1)] = min(abs(wf(:,1)-tim(1)));    % time Index
[delete timI(2)] = min(abs(wf(:,1)-tim(2)));
[delete timI(3)] = min(abs(wf(:,1)-tim(3)));
[delete timI(4)] = min(abs(wf(:,1)-tim(4)));

avrBack = mean(wf(timI(1):timI(2),2));           % averaged background [V]

area1 = sum(wf(timI(3):timI(4),2))*timStp;       % sum of signal [Vs]
areaB = avrBack*(tim(4)-tim(3));                 % background [Vs]
area2 = area1 - areaB;
chrg = area2*0.21*1e12;                       % charge [pC]

sigp2p = max(wf(timI(3):timI(4),2))-min(wf(timI(3):timI(4),2)); %peak-peak at signal
bgp2p = max(wf(timI(1):timI(2),2))-min(wf(timI(1):timI(2),2));  % peak-peak at bg
sn = sigp2p/bgp2p;                                              % s/n ratio
