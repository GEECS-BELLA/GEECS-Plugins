function fPathC = fDnnImgDir(scnNDir,scanNS,camClb)
%% return cell of the file path (no shot number) for Dnn magspec images
%
% fPathC = fBellaImgDir(scnNDir,scanNS)
%
% fPathC: cell of image file names (no shot number)
%
% scnNDir: scan# directory
% scanNS: scan number string
%

% Written by Kei Nakamura
% 2018/2/22 ver.1: created
% 2018/11/19 ver.1b: file name in calib.

%% main body

% diagC{1} = 'HTT-MagCam-01';
% diagC{2} = 'HTT-MagCam-02';
% diagC{3} = 'HTT-MagCam-03';
% diagC{4} = 'HTT-MagCam-04';
diagC{1} = char(camClb(1).name);
diagC{2} = char(camClb(2).name);
diagC{3} = char(camClb(3).name);
diagC{4} = char(camClb(4).name);

for i=1:4
    fPathC{i} = [scnNDir,filesep,diagC{i},filesep,'Scan',scanNS,'_',diagC{i},'_'];
end
