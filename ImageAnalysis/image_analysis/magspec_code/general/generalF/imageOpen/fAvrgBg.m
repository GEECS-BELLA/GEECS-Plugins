function bgImg = fAvrgBg(diagNam,bgScan,scnDir,anlDir)
%% give averaged bg after checking
%

%% Written by Kei Nakamura
% 2019/1/3 ver.1: created
%

%% main
bgNameL = dir([anlDir,filesep,'Scan*',diagNam,'_averaged.png']);

if isempty(bgNameL) % if no averaged bg
    bgScanNS = fGet3NmbStringV01(bgScan);   % scan# string for bg
    bgScnNDir = [scnDir,filesep,'Scan',bgScanNS];
    bgImg = fAvrgImg(bgScnNDir,diagNam);
    bgName = [anlDir,filesep,'Scan',bgScanNS,diagNam,'_averaged.png'];
    intImg = uint16(round(bgImg));  % integer image
    imwrite(intImg,bgName,'png');  % write integer image
else
    bgName = [anlDir,filesep,bgNameL(end).name];
    [bgImg,~] = f12bitPngOpnV04(bgName);
end
