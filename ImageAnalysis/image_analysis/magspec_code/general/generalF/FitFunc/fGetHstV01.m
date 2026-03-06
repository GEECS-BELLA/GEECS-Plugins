function hst = fGetHstV01(x,y)
% Create histogram like data for fitting functions
%
% hst = fGetHstV01(x,y)
% x: axis information (x axis)
% y: counts for each point (yaxis)
% hst: formatted for fitting function input
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get histogram, ver.1
%
% Kei Nakamura
% Created on Thursday, February 11th 2010.
% Modified on
%
% ver.1: created
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% main body
aaa = round(y);  % make it into integar to create histogram
ttlCnt = sum(aaa);  % total counts
hst = zeros(ttlCnt,1);
zz = 1;
for xx=1:numel(aaa);   % x
    while aaa(xx)>0
        hst(zz) = x(xx);
        zz = zz + 1;
        aaa(xx) = aaa(xx)-1;
    end
end
