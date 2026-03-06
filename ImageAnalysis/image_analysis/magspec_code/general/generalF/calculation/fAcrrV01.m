function acr = fAcrrV01(y)

%% calculate auto-correlation (A-CRR) of y
% acr = fACrrV01(y)
%
% acr: auto-correlation
% y:
%

%% Written by Kei Nakamura
% 2013/10/1 ver.1: created
%
%% Used in
% bellaFrogPhaseFitV07
% bellaSsaSimV04

%% main body
aSize = numel(y);   % array size
acrP = zeros(1,aSize-1);   % acr padding
acr1 = [acrP,y,acrP,acrP]; % padded y
acr = y;    % allocation

for kk=1:aSize
    jj = kk + 0.5*aSize -1;
    indA1 = jj;
    indA2 = jj + 2*aSize - 2;
    acr(kk) = sum(acr1(indA1:indA2).*acr1(aSize:3*aSize-2));
end
