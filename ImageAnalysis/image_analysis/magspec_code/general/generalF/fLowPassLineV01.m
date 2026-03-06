function clmVctr = fLowPassLineV01(clmVctr,fct,min,itr)
%% return dispersions for CPA system
% dsprsn = fCpaDispersionV02(L,N,lmb0,alp,sgn,ord)
%
% dsprsn:

%% Written by Kei Nakamura
% 2015/9/16 ver.1: created
%

%% Used in

%% main
sN = clmVctr(1);    % start#
eN = clmVctr(end);  % end#
for ii = 1:itr
    clm1 = [sN;clmVctr;eN];
    clm2 = 0.5*([clmVctr;eN;eN] + [sN;sN;clmVctr]);
    lgcX = clm1>clm2*fct & clm1>min; % logic for xray hit
    lgcOK = ~lgcX;  % logic for ok.
    clmOut = clm1.*lgcOK + clm2.*lgcX;
    clmVctr = clmOut(2:end-1);
end

for ii = 1:itr
    clm1 = [sN;sN;clmVctr;eN;eN];
    clm2 = 0.5*([clmVctr;eN;eN;eN;eN] + [sN;sN;sN;sN;clmVctr]);
    lgcX = clm1>clm2*fct & clm1>min; % logic for xray hit
    lgcOK = ~lgcX;  % logic for ok.
    clmOut = clm1.*lgcOK + clm2.*lgcX;
    clmVctr = clmOut(3:end-2);
end
