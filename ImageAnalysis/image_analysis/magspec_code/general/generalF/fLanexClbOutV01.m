function lanexClb = fLanexClbOutV01(lanexTtl,lanexData,setN)
%% return LANEX calibration information for specific set#
%
% lanexClb = fLanexClbOutV01(lanexTtl,lanexData,setN)
%
% lanexClb: structure containing lanex calibration info
%   .fovSlp: fov slope
%   .fovOffst: FOV offset
%   .vgnt4: vignette 4th order coeff.
%   .vgnt2: vignette 2nd order coeff.
%   .vgnt0: vignette 0th order coeff.
%   .sense2: sensitivity 2nd order coeff.
%   .sense1: sensitivity 1st order coeff.
%   .sense0: sensitivity oth order coeff.
%   .width: full width of the camera pixel
%   .height: full height of the camera pixe
%
% lanexTtl,lanexData:
%
% setN: set number

%% Written by Kei Nakamura
% 2013/3/20 ver.1:, created

%% main body
lanexClb.fovSlp = str2double(lanexData(setN,fLogClmnFindV01(lanexTtl,'FOV slope'))); % fov column
lanexClb.fovOffst = str2double(lanexData(setN,fLogClmnFindV01(lanexTtl,'FOV offset'))); % fov column
lanexClb.vgnt4 = str2double(lanexData(setN,fLogClmnFindV01(lanexTtl,'vignette 4'))); % fov column
lanexClb.vgnt2 = str2double(lanexData(setN,fLogClmnFindV01(lanexTtl,'vignette 2'))); % fov column
lanexClb.vgnt0 = str2double(lanexData(setN,fLogClmnFindV01(lanexTtl,'vignette 0'))); % fov column
lanexClb.sense2 = str2double(lanexData(setN,fLogClmnFindV01(lanexTtl,'sensitivity 2'))); % fov column
lanexClb.sense1 = str2double(lanexData(setN,fLogClmnFindV01(lanexTtl,'sensitivity 1'))); % fov column
lanexClb.sense0 = str2double(lanexData(setN,fLogClmnFindV01(lanexTtl,'sensitivity 0'))); % fov column
lanexClb.width = str2double(lanexData(setN,fLogClmnFindV01(lanexTtl,'full width'))); % fov column
lanexClb.height = str2double(lanexData(setN,fLogClmnFindV01(lanexTtl,'full height'))); % fov column
