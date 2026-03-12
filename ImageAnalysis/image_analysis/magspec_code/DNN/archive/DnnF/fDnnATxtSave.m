function ok = fDnnATxtSave(yA,img,txtA,namT,ttlA)
%% anglar disctibution info output for bella magspec
%
% ok = fBellaATxtSaveV01(yA,img,txtA,namT,ttlA)
%
% yA: y Axis info
% img: image
% txtA: empty array for output
% namT: save path
% ttlA: labels


%% Written by Kei Nakamura
% 2013/3/25 ver.1: created

%% main body
txtA(1,:) = yA.angl;     % x-axis [mrad]
txtA(2,:) = 0.001*sum(img,2);   % [pC]
txtA(3,:) = txtA(2,:)/yA.da; % [pC/mrad]

[nmbDOutA,~] = size(txtA);
ok = fTxtOutV01(namT,nmbDOutA,ttlA,txtA');
