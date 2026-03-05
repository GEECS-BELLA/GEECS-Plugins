function ok = fDnnETxtSave(xA,img,fld,txtE,namT,ttlE,x)
%% return uniform angle and momentum, combined images for bella magspec cameras
%
% uaImgC = fBellaUaV01(imgC,x,anglC)
%
% uaImgC: uniform angle image cell
% xA: x angle axix info, stiched
%
% imgC: image cell
% anglC: angle Cell
% dAngleC: dAngle cell
% y: structure with y info
%   .angl: angle [mrad]

%% Written by Kei Nakamura
% 2018/3/13 ver.1: created
% 2018/5/1 ver.1b: gap momemtum

%% main body
txtE(1,:) = fld*xA.mmt;     % x-axis [MeV]
txtE(2,:) = 0.001*sum(img);   % [pC]
txtE(3,:) = txtE(2,:)/(xA.dp*fld); % [pC/MeV]
txtE(4,:) = xA.accp;           % acceptance [mrad]
txtE(5,:) = xA.mmt;     % nrm.x-axis [MeV/c/T]
txtE(6,:) = 0;
txtE(6,1) = x(2).mmt(1)*fld;    % gap mmt
txtE(6,end) = x(3).mmt(end)*fld;    % gap mmt
[nmbDOutE,~] = size(txtE);
ok = fTxtOutV01(namT,nmbDOutE,ttlE,txtE',8);
