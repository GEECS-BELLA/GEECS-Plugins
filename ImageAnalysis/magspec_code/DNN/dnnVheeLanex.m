%% This program reads VHEE Lanex camera image and analyze
% PROGRAM dnnVheeLanex.m
% HTT VHEE Lanex analysis (real-time)

%% Written by Kei Nakamura
% 2026/2/17 ver.1: initial
%

%% initializations
clear all; %close all; warning off all;
fig1 = figure(1);
set(gcf,'color',[1 1 1]);
set(fig1, 'Position', [23 450 900 400]);
fig2 = figure(2);
set(gcf,'color',[1 1 1]);
set(fig2, 'Position', [23 56 900 300]);

%% input parameters
day = '26_0227';    % day string to locate data to analyze
mode = 1;   % mode (1: server, 2: local)
addScan = 1; % 1 to add summary to sfile
autoM = 1;  % 1 for auto mode (real-time analysis)
scan = [54]; % for auto mode, first scan
bgScan = 4;
roiRmrad = 0.5; % radius for charge-sum ROI [mm]
roiXY = [2,-3.8]; % roi location [mrad]
nc2Gy = 0.97*12.12/0.185; % nC to dose factor (depends on roiRMrad)

mgspcA = 10; % magspec acceptance in radius [mrad]
src2LnxM = 1.500; % source to lanex in m
fntSz = 12; % font size for fig
viewRoi = [-20 20 -20 20]; % roi for figure [mrad]

%% inpute parameters (not frequently changed)
% if autoM==1
%     scan = scan:1000;
% end
fctX = 2; pitX = 1; minX = 1; itr = 5;    % pit size, factor, and min. counts for low pass filter
nmbDOutL = 8;                                  % number of data out for main log
nmbDOutV = 4;                                   % number of data out for angle txt
[ttlL,ttlE,ttlA,ttlP,lbl,ttlV,ttlVS] = fDnnLabel;           % for label

%% directories, misc
[rDir,clbDir,scnDir,anlDir] = fDnnRootDir(day,mode);  % mother directory
if ~exist(anlDir,"dir")
    mkdir(anlDir);
end
lPass = [fctX pitX minX itr];                      % for low pass
nmbScan = numel(scan);  % number of scans to go through

%% prep calibration information
camClbPath = fPickCalib(clbDir,day,'DnnVarianCam');
lanexClbPath = fPickCalib(clbDir,day,'lanexCalib');
[camTtl,camData,~,~,~] = fLogReadV07(camClbPath);     % read calibration file
camClb = fDnnCamClb(camTtl,camData);     % extract camera calibration date
[lanexTtl,lanexData,~,~,~] = fLogReadV07(lanexClbPath); % load lanex calibration
lanexClb = fLanexClbOutV01(lanexTtl,lanexData,camClb(5).set); % extract vhee lanex
[c2c,vgntC] = fDnnLanex(camClb,lanexTtl,lanexData);   % counts to charge (c2c), vignette matrix

%% e-beam prf axis information
tmpA = camClb(end).fov/lanexClb.width; % mm/pixel
% x axis in mm (left pos is side center)
xAxMM = tmpA*([1:camClb(5).width])+camClb(5).leftPos;
xAxMM = xAxMM(camClb(5).xSt:camClb(5).xEd);
% y axis in mm
%yAxMM = tmpA*([1:camClb(5).height]-camClb(5).yCntr+camClb(5).ySt);
yAxMM = tmpA*([1:camClb(5).height]-camClb(5).yCntr);
yAxMM = -yAxMM(camClb(end).ySt:camClb(end).yEd);
%
xAxMrad = xAxMM/src2LnxM;  % x axis in mrad
yAxMrad = yAxMM/src2LnxM;  % y axis in mrad, flipped for convenience
dmrad = xAxMrad(2) - xAxMrad(1);

% roi (may introduce logic here)
radD = [0:0.01:2*pi];   % radian for drawing
[xMesh,yMesh] = meshgrid(xAxMrad,yAxMrad);
rMesh = sqrt((xMesh-roiXY(1)).^2+(yMesh-roiXY(2)).^2);% r mesh [mrad]
rLgc = rMesh<=roiRmrad;% ROI logic
mgsLgc = rMesh<=mgspcA; % magspec acceptance logic

%% read background images
bgImgNam = [anlDir,filesep,'Scan*',char(camClb(5).name),'_averaged.png'];
bgNameL = dir(bgImgNam);
if ~isempty(bgNameL) % background exist, load
    bgImg = f12bitPngOpnV04([bgNameL(1).folder filesep bgNameL(1).name]);
else % background not exist, make
    scanNS = fGet3NmbStringV01(bgScan);  % 3 digit scan number string
    scnDiagDir = [scnDir,filesep,'Scan',scanNS,filesep,char(camClb(5).name)]; % diag folder
    list1 = dir([scnDiagDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'*.png']);
    for jj = 1:numel(list1)
        imgPath = [list1(jj).folder filesep list1(jj).name];
        if jj==1
            bgImg = f12bitPngOpnV04(imgPath);
        else
            bgImg = bgImg + f12bitPngOpnV04(imgPath);
        end
    end
    bgImg = bgImg/numel(list1);
    bgImg = bgImg(camClb(5).ySt:camClb(5).yEd,camClb(5).xSt:camClb(5).xEd); % ROI
    bgImgNam = [anlDir,filesep,'Scan',scanNS,char(camClb(5).name),'_averaged.png'];
    intImg = uint16(round(bgImg));  % integer image
    imwrite(intImg,bgImgNam,'png');  % write integer image
end
% allocation
rawImg = zeros(camClb(5).height,camClb(5).width); % raw image for allocation
img = bgImg; acImg = img*0;

%% scan loop
disp('scan shot charge[pC]');
ss = 1;     % index for scan
scanN = scan(1);     % index for scan-while
shotN = 1;      % shot number

while scanN>0 % scan loop

    %% directoris
    accChgNc = 0; accChgRNc = 0; % accumulated charge reset
    scanNS = fGet3NmbStringV01(scanN);  % 3 digit scan number string
    scnNDir = [scnDir,filesep,'scan',scanNS];            % scan# directory
    scnDiagDir = [scnDir,filesep,'Scan',scanNS,filesep,char(camClb(5).name)]; % scan diag dir

    anlNDir = [anlDir,filesep,'scan',scanNS];        % analysis-scan directory
    anlDiagDir = [anlNDir,filesep,char(camClb(5).name)];       % directory for analyzed img
    anlPptDir = [anlNDir,filesep,char(camClb(5).name),'ppt'];       % directory for analyzed img

    logPath = [scnNDir,filesep,'ScanDataScan',scanNS,'.txt'];   % log file path

    firstShot = [scnDiagDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'_001.png'];

    %% wait for the first shot to proceed
    if autoM==1
        while ~exist(firstShot,'file')
            disp('control-c to stop program')
            pause(10)
        end
        nmbSht = 99999;  % number of shot for live mode
    else % if not live, read log to get nmbSht
        [logTtl,logData,~,nmbSht,~] = fLogReadV07(logPath);
    end

    %% make folders to proceed
    if ~exist(anlNDir,'dir')
        mkdir(anlNDir);
    end
    mkdir(anlDiagDir); mkdir(anlPptDir);

    %% loop for shots
    while shotN<=nmbSht % loop goes on till exceeds nmb of shot

        %% initializatin, read and process image
        sNmbS = fGet3NmbStringV01(shotN);  % 3 digit shot number string
        imgPath = [scnDiagDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'_',sNmbS,'.png'];   % complete file path

        % wait for shot>2
        if autoM==1
            while ~exist(imgPath,'file')
                disp('waiting for next shot, control-c to stop program')
                pause(1)
                if exist(logPath,'file')
                    break
                end
            end
        end

        % no next shot image but scan file exist, break
        if ~exist(imgPath,'file')&& exist(logPath,'file')
            break
        end

        % Read
        [rawImg(:,:),~] = f12bitPngOpnV04(imgPath);
        % ROI
        img(:,:) = rawImg(camClb(5).ySt:camClb(5).yEd,camClb(5).xSt:camClb(5).xEd);
        % BG
        img(:,:) = fTrexTonyBgV01(img,bgImg);   % back ground subtraction
        % low pass
        [img(:,:),~] = fXrayOutV10(img,lPass);
        % vignette
        img(:,:) = img.*vgntC{5};
        % charge conversion
        img(:,:) = c2c(5)*img; % [fC]
        acImg(:,:) = acImg + img; % accumulated image [fC]

        %% save output for each shot
        % x-axis
        namT = [anlDiagDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'X_',sNmbS,'.txt'];
        ok = fTxtOutV01(namT,nmbDOutV,ttlV,[xAxMM',xAxMrad',0.001*sum(img)',(0.001*sum(img)/dmrad)']);
        % y-axis
        namT = [anlDiagDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'Y_',sNmbS,'.txt'];
        ok = fTxtOutV01(namT,nmbDOutV,ttlV,[yAxMM',yAxMrad',0.001*sum(img,2),0.001*sum(img,2)/dmrad]);
        % image [fC]
        namT = [anlDiagDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'_fC_',sNmbS,'.png'];
        intImg = uint16(round(img));  % integer image [fC]
        imwrite(intImg,namT,'png');  % write integer image

        %% analysis and data
        chgPc = 1e-3*sum(sum(img)); % charge [pC]
        chgPcR = 1e-3*sum(sum(img.*rLgc)); % ROI-charge [pC]
        chgPcMg = 1e-3*sum(sum(img.*mgsLgc)); % magspec charge [pC]

        accChgNc = accChgNc + chgPc/1000; % accumulated charge [nC]
        accChgRNc = accChgRNc + chgPcR/1000; % accumulated charge-roi [nC]

        spot = fSpotAnalysisV02(xAxMrad,yAxMrad,img,1);
        % spot  1-2: peak location (x,y)
        %       3-4: mean location (x,y)
        %       5-6: mean with threshold (x,y)
        %       7-8: FWHM (x,y, from inside)
        %       9-10: std (standard deviation, x,y)
        %       11-12: 1/e^2 (x,y)

        dat(shotN,1) = shotN;  % shotnumber
        dat(shotN,2) = chgPc;  % charge [pC]
        dat(shotN,3) = chgPcMg;  % magspec (10mrad) charge [pC]
        dat(shotN,4) = chgPcR;  % roi-charge [pC]
        dat(shotN,5) = spot(1);  % peakX [mrad]
        dat(shotN,6) = spot(2);  % peakY [mrad]
        dat(shotN,7) = spot(7);  % fwhmX [mrad]
        dat(shotN,8) = spot(8);  % fwhmY [mrad]

        info{1} = ['Scan ',num2str(scanN),', Shot ',num2str(shotN)];
        info{2} = ['charge: ' num2str(chgPc,3) ' pC'];
        info{3} = ['10mrad charge: ' num2str(chgPcMg,3) ' pC'];
        info{4} = ['roi-charge: ' num2str(chgPcR,3) ' pC'];
        info{5} = ['Acc chrg: ' num2str(accChgNc,4) ' nC'];
        info{6} = ['roi-Acc chrg: ' num2str(accChgRNc,4) ' nC'];

        %% figure
        set(0,'CurrentFigure',fig1)
        subplot(2,3,2)
        pcolor(xAxMrad,yAxMrad,img)
        hold on
        plot([xAxMrad(1) xAxMrad(end)],[0 0],'k--')
        plot([0 0],[yAxMrad(1) yAxMrad(end)],'k--')
        plot(roiRmrad*cos(radD)+roiXY(1),roiRmrad*sin(radD)+roiXY(2),'w--')
        plot(mgspcA*cos(radD),mgspcA*sin(radD),'w--')
        plot(spot(1),spot(2),'rx','MarkerSize', 10)
        hold off
        shading interp
        %colorbar
        axis(viewRoi)
        aa = axis;
        set(gca,'fontsize',fntSz);

        subplot(2,3,1)
        plot(sum(img,2),yAxMrad)
        bb = axis;
        axis([bb(1:2) aa(3:4)])
        grid on
        ylabel('mrad')
        set(gca,'fontsize',fntSz);

        subplot(2,3,5)
        plot(xAxMrad,sum(img))
        bb = axis;
        axis([aa(1:2) bb(3:4)])
        grid on
        xlabel('mrad')
        set(gca,'fontsize',fntSz);

        fPltInfV02(subplot(2,3,4),info,6,fntSz);

        subplot(1,3,3)
        doseAR = 1e-6*sum(sum(acImg.*rLgc))*nc2Gy; % ROI-dose [Gy]
        doseTx = ['roi-dose: ' num2str(doseAR,4) ' Gy'];
        dsRate = doseAR/shotN; % dose rate [Gy/shot]
        drTx = ['dose rate: ' num2str(1000*dsRate,4) ' mGy/shot'];

        pcolor(xAxMrad,yAxMrad,acImg)
        hold on
        plot([xAxMrad(1) xAxMrad(end)],[0 0],'k--')
        plot([0 0],[yAxMrad(1) yAxMrad(end)],'k--')
        plot(roiRmrad*cos(radD)+roiXY(1),roiRmrad*sin(radD)+roiXY(2),'k--')
        %plot(mgspcA*cos(radD),mgspcA*sin(radD),'w--')
        %plot(spotA(1),spotA(2),'rx','MarkerSize', 10)
        text(-5,6,doseTx,'FontSize',fntSz,'Color','w');
        text(-5,5,drTx,'FontSize',fntSz,'Color','w');
        %text(-19,18,'High','FontSize',14,'Color','w');
        %text(-19,0,'PW','FontSize',14,'Color','w');
        %text(13,0,'LCave','FontSize',14,'Color','w');
        hold off
        shading interp
        %colorbar
        axis([-11/2 11/2 -15/2 15/2])
        xlabel('mrad'); ylabel('mrad');
        aa = axis;
        set(gca,'fontsize',fntSz);

        drawnow
        namOut = [anlPptDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'ppt_',sNmbS,'.png'];
        saveSameSize(fig1,'format','png','renderer','opengl','file',namOut);

        %% test fig
        % figure(100)
        % pcolor(xAxMM,yAxMM,img)
        % shading interp
        % hold on
        % plot([xAxMM(1) xAxMM(end)],[0 0],'r--')
        % plot([0 0],[yAxMM(1) yAxMM(end)],'r--')
        % plot(roiRmrad*cos(radD),roiRmrad*sin(radD),'w--')
        % % plot([0 1000],[-22. -22.])
        % hold off
        % colorbar
        % axis equal
        % aa = axis;

        %% display
        s1 = num2str(0.1*round(10*chgPc));
        disp([scanNS,',',sNmbS,',',s1,'pC'])

        %% sFile check for real-time mode (while break)
        shotN = shotN + 1;
        sFileL = exist(logPath,'file'); % s-file exist logic
        if sFileL
            [~,~,~,nmbSht] = fLogReadV07(logPath);
        end

    end % shots loop

    %% accumulated image
    spotA = fSpotAnalysisV02(xAxMrad,yAxMrad,acImg,1);
    chgA = sum(sum(acImg))/1e6;
    chgAR = 1e-6*sum(sum(acImg.*rLgc)); % ROI-charge [nC]
    chgAMg = 1e-6*sum(sum(acImg.*mgsLgc)); % magspec charge [nC]

    infoA{1} = [date ' Scan ',num2str(scanN)];
    infoA{2} = ['accm. charge: ' num2str(chgA,3) ' nC'];
    infoA{3} = ['10mrad charge: ' num2str(chgAMg,3) ' nC'];
    infoA{4} = ['roi-charge: ' num2str(chgAR,3) ' nC'];
    infoA{5} = ['xAngle: ' num2str(spotA(1),4) ' mrad'];
    infoA{6} = ['yAngle: ' num2str(spotA(2),4) ' mrad'];

    set(0,'CurrentFigure',fig2)
    subplot(1,3,2)
    pcolor(xAxMrad,yAxMrad,acImg)
    hold on
    plot([xAxMrad(1) xAxMrad(end)],[0 0],'k--')
    plot([0 0],[yAxMrad(1) yAxMrad(end)],'k--')
    plot(roiRmrad*cos(radD)+roiXY(1),roiRmrad*sin(radD)+roiXY(2),'k--')
    plot(mgspcA*cos(radD),mgspcA*sin(radD),'w--')
    plot(spotA(1),spotA(2),'rx','MarkerSize', 10)
    text(-19,18,'High','FontSize',14,'Color','w');
    text(-19,0,'PW','FontSize',14,'Color','w');
    text(13,0,'LCave','FontSize',14,'Color','w');
    hold off
    shading interp
    %colorbar
    axis(viewRoi)
    xlabel('mrad'); ylabel('mrad');
    aa = axis;
    set(gca,'fontsize',fntSz);

    fPltInfV02(subplot(1,3,1),infoA,6,fntSz);

    %% save output for a scan
    namOut = [anlNDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'Log.txt'];
    ok = fTxtOutV01(namOut,nmbDOutL,ttlVS,dat);
    sPath = [anlDir,filesep,'s',num2str(scanN),'.txt'];   % sfile path
    if addScan==1
        pause(30)
        [logTtl1,logDat1] = fAdd2LogFile(sPath,ttlVS(2:end),dat(:,2:end));
        ictC = str2double(logDat1(:,fLogClmnFindV01(logTtl1,'charge_pC'))); % ict charge
        p1 = polyfit(ictC,dat(:,2),1);
        inf1 = ['slope: ' num2str(p1(1),4)];
        inf2 = ['offset: ' num2str(p1(2),4),' pC'];

        subplot(1,3,3)
        plot(ictC,dat(:,2),'o')
        aa = axis;
        hold on
        plot([0,250], polyval(p1,[0,250]))
        text(aa(1)*1.1, aa(4)*0.95,inf1,'FontSize',fntSz)
        text(aa(1)*1.1, aa(4)*0.9,inf2,'FontSize',fntSz)
        hold off
        axis(aa);
        set(gca,'fontsize',fntSz);

    end

    %% other saving
    % ppt img
    drawnow
    namOut = [anlNDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'_accum.png'];
    saveSameSize(fig2,'format','png','renderer','opengl','file',namOut);
    % x-axis
    namT = [anlNDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'Xaccum.txt'];
    ok = fTxtOutV01(namT,nmbDOutV,ttlV,[xAxMM',xAxMrad',0.001*sum(acImg)',(0.001*sum(acImg)/dmrad)']);
    % y-axis
    namT = [anlNDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'Yaccum.txt'];
    ok = fTxtOutV01(namT,nmbDOutV,ttlV,[yAxMM',yAxMrad',0.001*sum(acImg,2),0.001*sum(acImg,2)/dmrad]);
    % image [fC]
    namT = [anlNDir,filesep,'Scan',scanNS,'_',char(camClb(5).name),'accum_fC_.png'];
    intImg = uint16(round(acImg));  % integer image [fC]
    imwrite(intImg,namT,'png');  % write integer image



    %% reset
    shotN = 1; % reset shot number
    dat = 0; % reset dat
    acImg(:,:) = acImg*0; % accumulated image reset
    disp('scan done')

    %% go next scan
    if autoM==1
        scanN = scanN + 1;
    else
        ss = ss + 1;
        if numel(scan)<ss
            scanN = -1;% to end scan loop
        else
        scanN = scan(ss);
        end
    end

end % scan loop
