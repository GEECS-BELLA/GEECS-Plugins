%% This program compares VHEE Lanex and RCF
% PROGRAM vheeLanesRcf.m
% HTT VHEE Lanex RCF comparison

%% Written by Kei Nakamura
% 2026/2/28 ver.1: initial
%

%% initializations
clear all; %close all; warning off all;
fig1 = figure(1);
set(gcf,'color',[1 1 1]);
set(fig1, 'Position', [550 500 1800 600]);
% fig2 = figure(2);
% set(gcf,'color',[1 1 1]);
% set(fig2, 'Position', [23 56 900 300]);

%% input parameters
day = '26_0224';    % day string to locate data to analyze
scan = [34]; % for auto mode, first scan
rcf = 167;  % rcf#
rcfRoi = [1,254,10,185]; % yRoi - xRoi
lnxRoi = [80,180,70,185]; % yRoi - xRoi
mode = 1;   % mode (1: server, 2: local)
roiRmm = 2; % radius for charge-sum ROI [mm]
nc2Gy = 0.97*12.12/0.185; % nC to dose factor (depends on roiRMrad)

mgspcA = 10; % magspec acceptance in radius [mrad]
src2LnxM = 1.500; % source to lanex [m]
src2Smpl = 1.580;
fntSz = 12; % font size for fig
viewRoi = [-15 25 -20 15]; % roi for figure [mrad]

%% inpute parameters (not frequently changed)
fctX = 1.3; pitX = 2; minX = 1; itr = 2;    % pit size, factor, and min. counts for low pass filter
nmbDOutL = 8;                                  % number of data out for main log
nmbDOutV = 4;                                   % number of data out for angle txt
[ttlL,ttlE,ttlA,ttlP,lbl,ttlV,ttlVS] = fDnnLabel;           % for label

%% directories, misc
[rDir,clbDir,scnDir,anlDir] = fDnnRootDir(day,mode);  % mother directory
lPass = [fctX pitX minX itr];                      % for low pass
radD = [0:0.01:2*pi];

%% directoris
scanNS = fGet3NmbStringV01(scan);  % 3 digit scan number string
scnNDir = [scnDir,filesep,'scan',scanNS];            % scan# directory
scnDiagDir = [scnDir,filesep,'Scan',scanNS,filesep,'HTT-VHEE_Lanex']; % scan diag dir

anlNDir = [anlDir,filesep,'scan',scanNS];        % analysis-scan directory
anlDiagDir = [anlNDir,filesep,'HTT-VHEE_Lanex'];       % directory for analyzed img
anlPptDir = [anlNDir,filesep,'HTT-VHEE_Lanex','ppt'];       % directory for analyzed img

imgPath = [anlNDir,filesep,'Scan',scanNS,'_HTT-VHEE_Lanexaccum_fC_.png']; % image path
xPath = [anlNDir,filesep,'Scan',scanNS,'_HTT-VHEE_LanexXaccum.txt'];
yPath = [anlNDir,filesep,'Scan',scanNS,'_HTT-VHEE_LanexYaccum.txt'];

logPath = [scnNDir,filesep,'ScanDataScan',scanNS,'.txt'];   % log file path

%rcfD = ['/Users/KNakamura/Library/CloudStorage/GoogleDrive-knakamura@lbl.gov/Shared drives/BELLA PW Team/RCF/20260223_EBT3_HTT'];
%rcfND = [rcfD filesep num2str(rcf)]; % rcf# dir
rcfND = [anlDir,filesep,'RCF'];
rcfPath = [rcfND filesep 'Rcf' num2str(rcf) '_200dpi_mGy.png'];

%% lanex read
[xTtl,xData,~,~,~] = fLogReadV07(xPath);
x.mrad = str2double(xData(:,fLogClmnFindV01(xTtl,'Angle_mrad'))); %
[yTtl,yData,~,~,~] = fLogReadV07(yPath);
y.mrad = str2double(yData(:,fLogClmnFindV01(yTtl,'Angle_mrad'))); %
img = double(imread(imgPath));

img = img(lnxRoi(1):lnxRoi(2),lnxRoi(3):lnxRoi(4));
x.mrad = x.mrad(lnxRoi(3):lnxRoi(4));
y.mrad = y.mrad(lnxRoi(1):lnxRoi(2));

spotA = fSpotAnalysisV02(x.mrad',y.mrad',img,1);

% projection at sample z
x.mm = src2Smpl*x.mrad; y.mm = src2Smpl*y.mrad;
dmm = x.mm(2) - x.mm(1); % dx [mm]
img = 0.001*img/dmm^2; % charge density [pC/mm^2]
spotmm = fSpotAnalysisV02(x.mm',y.mm',img,0);
xPosL = 0.5*(spotmm(1)+spotmm(3)); % x position
yPosL = 0.5*(spotmm(2)+spotmm(4)); % x position
peakCD = max(max(img)); %peak charge density [pC/mm^2]
[xMesh,yMesh] = meshgrid(x.mm,y.mm);
xMesh = xMesh - xPosL;
yMesh = yMesh - yPosL;
rMesh = sqrt(xMesh.^2 + yMesh.^2);
rLgc = rMesh<=roiRmm; % radius logic
meanCD = mean(img(rLgc)); % roi-mean charge density [pc/mm^2]

% info
infoA{1} = [day ' Scan ',num2str(scan)];
infoA{2} = ['mx CD: ' num2str(peakCD,4) ' pc/mm^2'];
infoA{3} = ['mean CD: ' num2str(meanCD,4) ' pc/mm^2'];
infoA{4} = ['xPos: ' num2str(xPosL,4) ' mm'];
infoA{5} = ['yPos: ' num2str(yPosL,4) ' mm'];
infoA{6} = ['xFWHM: ' num2str(spotmm(7),4) ' mm'];
infoA{7} = ['yFWHM: ' num2str(spotmm(8),4) ' mm'];

%% rcf read
imgR = double(imread(rcfPath))/1000; %'[Gy]'
imgR = fXrayOutV10(imgR,[1.5,1,1,1]);
imgR = fXrayOutV10(imgR,lPass);
imgR = fliplr(imgR);
imgR = imgR(rcfRoi(1):rcfRoi(2),rcfRoi(3):rcfRoi(4));
rcfSz = size(imgR);
rcfX = 25.4*([1:rcfSz(2)]-rcfSz(2)/2)/200; % 200 dpi
rcfY = 25.4*([1:rcfSz(1)]-rcfSz(1)/2)/200; % 200 dpi

spotR = fSpotAnalysisV02(rcfX,rcfY,imgR,0);
peakGy = max(max(imgR)); %peak charge density [pC/mm^2]

[xMesh2,yMesh2] = meshgrid(rcfX,rcfY);
xPosR = 0.5*(spotR(1)+spotR(3));
yPosR = 0.5*(spotR(2)+spotR(4));
xMesh2 = xMesh2 - xPosR;
yMesh2 = yMesh2 - yPosR;
rMesh2 = sqrt(xMesh2.^2 + yMesh2.^2);
rLgc2 = rMesh2<=roiRmm; % radius logic
meanGy = mean(imgR(rLgc2)); % roi-mean dose [Gy]

tx2 = [num2str(meanCD/meanGy,4),' pc/mm^2/Gy'];

% info
infoR{1} = ['RCF# ',num2str(rcf)];
infoR{2} = ['mx D: ' num2str(peakGy,4) ' Gy'];
infoR{3} = ['mean D: ' num2str(meanGy,4) ' Gy'];
infoR{4} = ['xPos: ' num2str(xPosR,4) ' mm'];
infoR{5} = ['yPos: ' num2str(yPosR,4) ' mm'];
infoR{6} = ['xFWHM: ' num2str(spotR(7),4) ' mm'];
infoR{7} = ['yFWHM: ' num2str(spotR(8),4) ' mm'];

%% figure lanex
set(0,'CurrentFigure',fig1)

fPltInfV02(subplot(2,4,5),infoA,7,fntSz);

subplot(2,4,2)
pcolor(x.mm,y.mm,img)
hold on
plot([x.mm(1) x.mm(end)],[0 0],'k--')
plot([0 0],[y.mm(1) y.mm(end)],'k--')
plot(roiRmm*cos(radD)+xPosL,roiRmm*sin(radD)+yPosL,'r--')
plot(src2Smpl*mgspcA*cos(radD),src2Smpl*mgspcA*sin(radD),'w--')
plot(xPosL,yPosL,'rx','MarkerSize', 10)
hold off
shading interp
colorbar
axis(viewRoi)
% xlabel('mm');
% ylabel('mm');
title('ChargeDen [pC/mm^2]')
set(gca,'fontsize',fntSz);

% x projection
subplot(2,4,6)
plot(x.mm,sum(img))
bb = axis;
axis([viewRoi(1) viewRoi(2)*1.6 bb(3:4)])
grid on
xlabel('mm')
set(gca,'fontsize',fntSz);

% y projection
subplot(2,4,1)
plot(sum(img,2),y.mm)
bb = axis;
axis([bb(1:2) viewRoi(3:4)])
grid on
ylabel('mm')
set(gca,'fontsize',fntSz);

fPltInfV02(subplot(2,4,8),infoR,7,fntSz);

%% rcf
subplot(2,4,3)
pcolor(rcfX,rcfY,imgR)
hold on
%plot([x.mm(1) x.mm(end)],[0 0],'k--')
%plot([0 0],[y.mm(1) y.mm(end)],'k--')
plot(roiRmm*cos(radD)+xPosR,roiRmm*sin(radD)+yPosR,'r--')
%plot(src2Smpl*mgspcA*cos(radD),src2Smpl*mgspcA*sin(radD),'w--')
plot(xPosR,yPosR,'rx','MarkerSize', 10)
text(-10,10,tx2,'fontsize',fntSz,'color','w');
hold off
shading interp
colorbar
%axis(viewRoi)
% xlabel('mm');
% ylabel('mm');
title('Dose [Gy]')
set(gca,'fontsize',fntSz);

% x projection
subplot(2,4,7)
plot(rcfX,sum(imgR))
%bb = axis;
%axis([viewRoi(1) viewRoi(2)*1.6 bb(3:4)])
grid on
xlabel('mm')
set(gca,'fontsize',fntSz);

% y projection
subplot(2,4,4)
plot(sum(imgR,2),rcfY)
%bb = axis;
%axis([bb(1:2) viewRoi(3:4)])
grid on
ylabel('mm')
set(gca,'fontsize',fntSz);

%%
 namOut = [rcfND,filesep,'Scan',scanNS,'_RCF',num2str(rcf),'doseCal.png'];
 saveSameSize(fig1,'format','png','renderer','opengl','file',namOut);
