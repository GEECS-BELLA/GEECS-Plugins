function [t,f,cx,cy,fx,fy] = fFastProfilerRead(fPath,smplF,varargin)
%% read file for fast profiler


%% Written by Kei Nakamura
% 2020/3/26 ver.1:, created

%% main body

data = importdata(fPath,'\t');
tBias = 0;
if nargin == 3
    roiN = varargin{1};
    data = data(roiN(1):roiN(2),:);
    tBias = roiN(1)/smplF;
end
[datL,~] = size(data);  % data length


t = [0:datL-1]/smplF + tBias;   % time [s]
f = smplF*(0:datL/2)/datL;  % freq [Hz]

cx = data(:,1); % centroid x
cx = cx - mean(cx); % avg 0

cxFft = fft(cx);
cxFft = fftshift(cxFft);    % shift 0 freq center
cxFftP = abs(cxFft/datL); % power spectrum
fx = cxFftP(datL/2:end);

cy = data(:,2); % centroid x
cy = cy - mean(cy); % avg 0

cyFft = fft(cy);
cyFft = fftshift(cyFft);    % shift 0 freq center
cyFftP = abs(cyFft/datL); % power spectrum
fy = cyFftP(datL/2:end);
