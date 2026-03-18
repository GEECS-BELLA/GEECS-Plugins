function out = fSttDatV04(inpt)
%% returns statistical data for plot
%
% out = fSttDatV04(inpt)
%
% inpt: N(shot) x M(element) dimension array
% out = 4M dimension array, containing mean,stdv,(mean-min), and (max-mean)
% for input shots
%
%%%%%%%%%%
% Kei Nakamura
% Created on Friday, Octorber 28th 2005
% Modified on Friday, February 12th 2010
%
% ver.4: output for errorbar
%

%% main body

meanD = mean(inpt);    % mean of data
stdD = std(inpt);      % standard deviation of data
lowD = meanD - min(inpt);   % low side for errobar
highD = max(inpt) - meanD;      % high side for errorbar
out = [meanD,stdD,lowD,highD];  % output
