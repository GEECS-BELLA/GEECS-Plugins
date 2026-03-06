%%%%%%%%%% Function fWvFrmPrcAllV02.m %%%%%%%%%%
% Wave Form Process All ver.2
% Kei Nakamura
% Created on Friday, May 18th 2007.
% Modified on Tuesday, August 19th 2008.

% This function processes all the waveform files in a directory

% out: out=[sht chrg sn];
% pthD: path for directory
% tim: time for background evaluation (tim(1)-tim(2)) and
%       time for signal evaluation (tim(3)-time(4)).

% out: [shot numer, charge, s/n]
% sht: shot number
% chrg: charge [pC]
% sn: S/N ratio

% ver.2: for 2008 exp.

function out = fWvFrmPrcAllV02(pthD, tim)   % function header

cDir = pwd;                                 % current directory
cd(pthD)                                    % move to the directoy to analyze
wLst = dir('*waveform.txt');                % waveform List
wNum = numel(wLst);                         % waveform number
h = waitbar(0,'Analyzing waveform...');     % waitbar for fun

for ii=1:wNum
    sht = str2num(getPartStrV02(wLst(ii).name,'z','w'));        % shot, no 'z'
    out(sht,1) = sht;
    [chg,sn] = fWvFrmPrcV03(wLst(ii).name,tim);  % signal, charge[pC], S/N
    out(sht,2:3) = [chg(1),sn(1)];
    out(sht,4:5) = [chg(2),sn(2)];
    waitbar(ii/wNum);
end
close(h);
cd(cDir);
