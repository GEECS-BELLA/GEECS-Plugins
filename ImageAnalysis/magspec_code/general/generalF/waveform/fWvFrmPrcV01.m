%%%%%%%%%% Function fWvFrmPrcV01.m %%%%%%%%%%
% Wave Form Process ver.1
% Kei Nakamura
% Created on Tuesday, May 8th 2007.
% Modified on

% This function reads text file,
% which recorded waveform of ICT signal at ALS experiments,
% and return the charge.

% chrg: charge [pC]
% fNam: file name to open
% tim: time for background evaluation (tim(1)-tim(2)) and
%       time for signal evaluation (tim(3)-time(4)).

function chrg = fWvFrmPrcV01(fNam, tim)  %function header

wf = textread(fNam);                           % download waveform
timStp = wf(2,1)-wf(1,1);                       % time step

% find index for times
[delete timI(1)] = min(abs(wf(:,1)-tim(1)));    % time Index
[delete timI(2)] = min(abs(wf(:,1)-tim(2)));
[delete timI(3)] = min(abs(wf(:,1)-tim(3)));
[delete timI(4)] = min(abs(wf(:,1)-tim(4)));

avrBack = mean(wf(timI(1):timI(2),2));           % averaged background [V]

area1 = sum(wf(timI(3):timI(4),2))*timStp;       % sum of signal [Vs]
areaB = avrBack*(tim(2)-tim(1));                 % background [Vs]
area2 = area1 - areaB;
chrg = area2*0.21*1e12;                       % charge [pC]
