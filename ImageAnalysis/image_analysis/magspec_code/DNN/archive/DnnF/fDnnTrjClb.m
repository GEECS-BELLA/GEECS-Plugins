function [trjFClb,trjSClb] = fDnnTrjClb(trjTtl,trjData)
%% return front-side calibration separately
%
% [trjFClb,trjSClb] = fDnnTrjClb(trjTtl,trjData)
%
% trjFCalib,trjSCalib: trajectory front and side calibration in structure
%   .mmt: momentum [MeV/c]
%   .screen: screen position [mm]
%   .incAgl: electron incident angle to screen [deg]
%   .path: total path length [m]
%   .divFX: diverging factor for x
%   .divFY: diverging factor for y
%
% trjTtl,trjData: title and data for trajectory calibration
%

%% Written by Kei Nakamura
% 2018/2/8 ver.1: succeeded from fBellaTrjCalib
%

%% main body
sideL = str2double(trjData(:,fLogClmnFindV01(trjTtl,'side logic'))); % fov side logic
k = find(sideL==1,1,'first');

%%  trajectory frong calibration
trjFClb.mmt = str2double(trjData(1:k-1,fLogClmnFindV01(trjTtl,'momentum [MeV/c]'))); % momentum side
trjFClb.screen = 1000*str2double(trjData(1:k-1,fLogClmnFindV01(trjTtl,'front screen [m]'))); % screen side
trjFClb.incAgl = str2double(trjData(1:k-1,fLogClmnFindV01(trjTtl,'bending angle at screen [dgr]'))); % bending angle at the screen
trjFClb.path = str2double(trjData(1:k-1,fLogClmnFindV01(trjTtl,'total path [m]'))); % total path
trjFClb.divFX = str2double(trjData(1:k-1,fLogClmnFindV01(trjTtl,'x conv fct rms'))); % conversion factor
trjFClb.divFY = str2double(trjData(1:k-1,fLogClmnFindV01(trjTtl,'y conv fct rms'))); % conversion factor
trjFClb.rsl = str2double(trjData(1:k-1,fLogClmnFindV01(trjTtl,'momentum rsl [%/mrad]')));

%%  trajectory side calibration
trjSClb.mmt = str2double(trjData(k:end,fLogClmnFindV01(trjTtl,'momentum [MeV/c]'))); % momentum front
trjSClb.screen = 1000*str2double(trjData(k:end,fLogClmnFindV01(trjTtl,'side screen [m]'))); % screen front
trjSClb.incAgl = abs(str2double(trjData(k:end,fLogClmnFindV01(trjTtl,'bending angle at screen [dgr]')))-120); % bending angle at the screen
trjSClb.path = str2double(trjData(k:end,fLogClmnFindV01(trjTtl,'total path [m]'))); % total path
trjSClb.divFX = str2double(trjData(k:end,fLogClmnFindV01(trjTtl,'x conv fct rms'))); % conversion factor
trjSClb.divFY = str2double(trjData(k:end,fLogClmnFindV01(trjTtl,'y conv fct rms'))); % conversion factor
trjSClb.rsl = str2double(trjData(k:end,fLogClmnFindV01(trjTtl,'momentum rsl [%/mrad]')));
