function [meanX,stdX,meanY,stdY] = fTonyPlotV01(x,y,bin)
%% return average x, y, sigma
% [x,sigX,y,sigY] = fTonyPlotV01(x,y,bin)
%
% x, y: input
% bin: bin from s-file
% meanX, meanY, stdX, stdY: output

%% Written by Kei Nakamura
% 2017/3/24 ver.1: created
%

%% main
% allocation
% meanX = zeros(bin(end),1);
% meanY = zeros(bin(end),1);
% stdX = zeros(bin(end),1);
% stdY = zeros(bin(end),1);

jj = 1;
for ii = 1:bin(end)
    slct = bin==ii;
    if sum(slct)
        meanX(jj) = mean(x(slct));
        meanY(jj) = mean(y(slct));
        stdX(jj) = std(x(slct));
        stdY(jj) = std(y(slct));
        jj = jj + 1;
    end
end
