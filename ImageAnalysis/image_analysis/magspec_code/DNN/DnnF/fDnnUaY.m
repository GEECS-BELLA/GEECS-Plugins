function yA = fDnnUaY(anglC,aglRsl)
%% return uniform angle y axis for dnn magspec cameras
%
% yA = fBellaUaYV02(anglC,aglRsl)
%
% yA: uniform angle y axis structure
%
% anglC: angle Cell
% aglRsl: angle resolutions


%% Written by Kei Nakamura
% 2018/2/9 ver.1: created
%

%% main body

mx = zeros(1,4); mn = zeros(1,4);
for i=1:4
    mx(i) = max(max(anglC{i}));
    mn(i) = min(min(anglC{i}));
end

edgA(1) = min(mn);  % edge angle, min
edgA(2) = max(mx);  % edge angle, max

da = (edgA(2) - edgA(1))/(aglRsl - 1); % dAngle
yA.angl = edgA(1):da:edgA(2);         % y axis (angle)
yA.da = da;                     % y axis (dAngle)
