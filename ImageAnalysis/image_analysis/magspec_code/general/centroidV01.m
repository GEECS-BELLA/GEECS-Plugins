%% gent controid of the images
%
% PROGRAM centroidV01.m %%%%%%%%%%
%
% Kei Nakamura

% This program reads image file from CCD cameras,
% study images

%
% ver.01: Thursday, May 19th, 2011


%% initialization
clear all
close all
warning off all;
fig1 = figure(1);
set(gcf,'color',[1 1 1]);
set(fig1, 'Position', [-1200 2700 400 300]);
set(gca,'FontSize',14,'FontWeight','bold');

%% input
fctX = 2; pitX = 1; minX = 5; itr = 3;    % pit size, factor, and min. counts for x-ray filter
nmbDOut = 10;                                   % number of data out
thre = 40;

%%
cDir = pwd;
imgDir ='/Volumes/data/100TWdata/Y2011/05-May/11_0518/scans/M23VTop';
cd(imgDir);

imgLst = dir('./*.png');
nmbImg = numel(imgLst); % number of image

dat = zeros(nmbImg,nmbDOut);            % blank output data

for ii=1:nmbImg

    %% read and process
    [img,info] = f12bitPngOpnV04([imgDir,'/',imgLst(ii).name]);
    pImg = img.*(img>thre);
    [szy,szx] = size(img);

    %% analysis
    imgPk = max(max(pImg));         % peak count
    imgSm = sum(sum(pImg));
    [sig,sigS,fwhm,fwhmS] = fPhosImgAnlV01(pImg,[1:szx],[1:szy]);

    %% data
    dat(ii,1:2) = [imgSm,imgPk];   % sum, peak counts
    dat(ii,3:6) = [sigS(1:2),fwhmS(1:2)];   % meanX, meanY, peakX, peakY [mm]
    dat(ii,7:10) = [sigS(3:4),fwhmS(3:4)];   % rmsX, rmsY, fwhmX, fwhmY [mm]
    disp([ii,imgPk]);

    %% figure and image saving
%     figure(1)
%     set(gca,'FontSize',14,'FontWeight','bold');
%     pcolor(pImg);
%     shading interp; colorbar;
%     set(gca,'FontSize',14,'FontWeight','bold');
%     nam = ['Scan',scanNS,fName2,sNmbS,'.png'];
%     nam = [anlDir,'/',nam];
%     saveSameSize(fig1,'format','png','renderer','opengl','file',nam);

end

%% log file saving
ttlL{1} = 'sum';
ttlL{2} = 'peak';
ttlL{3} = 'meanX';
ttlL{4} = 'meanY';
ttlL{5} = 'peakX';
ttlL{6} = 'peakY';
ttlL{7} = 'rmsX';
ttlL{8} = 'rmsY';
ttlL{9} = 'fwhmX';
ttlL{10} = 'fwhmY';
namOut = [imgDir,'/log.txt'];      % name for output file
ok = fTxtOutV01(namOut,nmbDOut,ttlL,dat);

cd(cDir)
