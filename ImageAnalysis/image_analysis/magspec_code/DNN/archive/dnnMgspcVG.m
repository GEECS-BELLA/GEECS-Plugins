%% This program reads processed files for DNN varian Magnetic Spectrometer, output viewgraphs
% PROGRAM dnnMgspecVG.m
% DNN Magnetic Spectrometer view graphs

%% Written by Kei Nakamura
% 2018/3/15 ver.1: created
% 2018/5/1 ver.1b: camera set, screen, gap momentum

%% initializations
clear all; close all; warning off all;
fig1 = figure(1);
set(gcf,'color',[1 1 1]);
set(fig1, 'Position', [100 30 1000 400]);
fig2 = figure(2);
set(gcf,'color',[1 1 1]);
set(fig2, 'Position', [100 30 1000 400]);
fig3 = figure(3);
set(gcf,'color',[1 1 1]);
set(fig3, 'Position', [100 30 350 350]);

%% input parameters
day = '25_0925';    % day string to locate data to analyze
mode = 1;   % mode (1: server, 2: local)
%scan = input('inp2ut scan number like [1:4,8]: ');
scan = [30];%[5,8:10,15,16,25];
roiM = [0,150];   % roi momentum [MeV]
roiA = [-25,25];   % roi angle [mrad]
addScan = 1;        % add summary to s-file
figV = 'off';   % off for fast process, on to check, figure visibility

%% inpute parameters (not frequently changed), misc.
set(fig1, 'visible', figV);
set(fig2, 'visible', figV);
set(fig3, 'visible', figV);
nmbDOutL = 9;                                  % number of data out for main log
[ttlL,ttlE,ttlA,ttlP,lbl] = fDnnLabel;           % for label
[rDir,clbDir,scnDir,anlDir] = fDnnRootDir(day,mode);  % mother directory
if ismac
    fntSz = 14; % mac font size
else
    fntSz = 9;  % windows font size
end
img1 = zeros(128,1024); % allocation

%% scan loop
disp('scan shot charge[pC]');
ss = 1;     % index for scan
for scanN=scan    % scan loop

    %% directoris and log
    scanNS = fGet3NmbStringV01(scanN);  % 3 digit scan number string
    anlNDir = [anlDir,filesep,'scan',scanNS];        % analysis-scan directory
    highDir = [anlNDir,filesep,'MgspcHtt'];       % directory for processed images
    pptDir = [anlNDir,filesep,'MgspcHttPpt'];       % directory for ppt ready images
    mkdir(pptDir);
    logPath = [anlDir,filesep,'s',num2str(scanN),'.txt'];   % sfile path
    [logTtl,logData,~,nmbSht,~] = fLogReadV07(logPath);

    %% scan dependent allocation
    dat = zeros(nmbSht,nmbDOutL);            % blank output data
    dat(:,1) = scanN;
    dat(:,2) = 1:nmbSht;              % put scan and shot number
    wtrFllM = zeros(nmbSht,1024);   % water fall for mmt
    wtrFllA = zeros(nmbSht,128);    % water fall for angle

    %% loop for shots
    for jj=1:nmbSht

        %% initializatin, read and process image
        sNmbS = fGet3NmbStringV01(jj);  % 3 digit shot number string
        fNameImg = [highDir,filesep,'Scan',scanNS,'_MgspcHtt_',sNmbS,'.png'];
        fNameSpc = [highDir,filesep,'Scan',scanNS,'_MgspcHttSpec_',sNmbS,'.txt'];
        fNameDiv = [highDir,filesep,'Scan',scanNS,'_MgspcHttDiv_',sNmbS,'.txt'];
        img1(:,:) = double(imread(fNameImg));
        [xTtl,xData,~,~,~] = fLogReadV07(fNameSpc);
        x.mmt = str2double(xData(:,fLogClmnFindV01(xTtl,'Momentum_MeV/c'))); % momentum [MeV/c]
        x.chrg = str2double(xData(:,fLogClmnFindV01(xTtl,'ChargeDen_pC/MeV/c'))); % momentum [MeV/c]
        x.accp = str2double(xData(:,fLogClmnFindV01(xTtl,'acepAngle_mrad'))); % momentum [MeV/c]
        x.gap = str2double(xData(:,fLogClmnFindV01(xTtl,'gap Mmt_MeV/c'))); % gap momentum [MeV/c]
        gapM(1) = x.gap(1); gapM(2) = x.gap(end);   % gap mmt
        [yTtl,yData,~,~,~] = fLogReadV07(fNameDiv);
        y.angl = str2double(yData(:,fLogClmnFindV01(yTtl,'Angle_mrad'))); % momentum [MeV/c]
        y.chrg = str2double(yData(:,fLogClmnFindV01(yTtl,'ChargeDen_pC/mrad'))); % momentum [MeV/c]
        dp = x.mmt(2) - x.mmt(1);
        da = y.angl(2) - y.angl(1);
        wtrFllM(jj,:) = x.chrg;
        wtrFllA(jj,:) = y.chrg;

        %% ROI
        roiML = x.mmt>=roiM(1)&x.mmt<=roiM(2);  % roi mmt logic
        roiAL = y.angl>=roiA(1)&y.angl<=roiA(2);
        % roi
        imgR = img1(roiAL,roiML);
        x.mmtR = x.mmt(roiML);
        x.chrgR = x.chrg(roiML);
        y.anglR = y.angl(roiAL);
        y.chrgR = y.chrg(roiAL);

        %% analysis
        x.enrgy = 0.001*sum(imgR).*x.mmtR'; % uJ
        bmEnrgyMj = 0.001*sum(x.enrgy); % beam energy, mJ
        chrgPc = 0.001*sum(sum(imgR)); %pC
        [~,meanMmt] = fGetRmsV01(x.mmtR,x.chrgR);
        [mmtFwhm,~,fwhmInd,~,pkInd] = fGetFwhmV04(x.mmtR,x.chrgR);
        pkMmt = x.mmtR(pkInd);   % peak momentum
        [anglFwhm,~,aFwhmInd,~,aPkInd] = fGetFwhmV04(y.anglR,y.chrgR);

        info{1} = [day(1:2) day(4:end) ' scan ' num2str(scanN) ' shot ' num2str(jj)];
        info{2} = ['beam charge: ' num2str(chrgPc,4) ' pC'];
        info{3} = ['beam energy: ' num2str(bmEnrgyMj,4) ' mJ'];
        info{4} = ['mean mmt: ' num2str(meanMmt,3) ' MeV/c'];
        info{5} = ['peak mmt: ' num2str(pkMmt,3) ' MeV/c'];
        info{6} = ['fwhm mmt: ' num2str(mmtFwhm,3) ' MeV/c'];
        info{7} = ['peak angle: ' num2str(y.angl(aPkInd),3) ' mrad'];
        info{8} = ['fwhm div.: ' num2str(anglFwhm,3) ' mrad'];

        %% data and display
        dat(jj,3:4) = [chrgPc bmEnrgyMj];  % charge pC, energy mJ
        dat(jj,5:7) = [meanMmt pkMmt mmtFwhm];  % mean, peak, fwhm mmt
        dat(jj,8:9) = [y.angl(aPkInd) anglFwhm];  % peak, fwhm angle
        disp([scanNS,',',sNmbS])

        %% figure 1:
        set(0,'CurrentFigure',fig1)
        fPltInfV02(subplot(2,4,5),info,8,fntSz);

        subplot(2,4,2:4)
        %imgR = fXrayOutV10(imgR,[1.5,2,1,1]);
        pcolor(x.mmtR,y.anglR,imgR/(dp*da))
        shading interp
        hold on
        plot(x.mmt,x.accp,'k-','linewidth',2)
        plot(x.mmt,-x.accp,'k-','linewidth',2)
        fill([x.mmt' x.mmt(end) 0 x.mmt(1)],[x.accp' 30 30 x.accp(1)],'k')
        fill([x.mmt' x.mmt(end) 0 x.mmt(1)],[-x.accp' -30 -30 -x.accp(1)],'k')
        fill([gapM(2) gapM(1) gapM(1) gapM(2) gapM(2)],[roiA(1) roiA -roiA],'k')
        hold off
        axis([roiM roiA])
        colorbar
        title('Charge density [pC/MeV/mrad]')
        %xlabel('Momentum [MeV/c]')
        % ylabel('Angle [mrad]')
        set(gca,'fontsize',fntSz);

        subplot(2,4,6:8)
        plot(x.mmt,x.chrg,'linewidth',2)
        aa = axis;
        axis([roiM(1) roiM(2)*1.125 aa(3:4)])
        hold on
        plot(x.mmtR(pkInd),x.chrgR(pkInd),'rx')
        if fwhmInd~=0
            plot(x.mmtR(fwhmInd(1)),x.chrgR(fwhmInd(1)),'rx')
            plot(x.mmtR(fwhmInd(2)),x.chrgR(fwhmInd(2)),'rx')
        end
        fill([gapM(2) gapM(1) gapM(1) gapM(2) gapM(2)],[0 0 aa(4) aa(4) 0],'k')
        hold off
        xlabel('Momentum [MeV/c]')
        ylabel('[pC/MeV/c]')
        set(gca,'fontsize',fntSz);

        subplot(2,4,1)
        plot(y.chrgR,y.anglR)
        aa = axis;
        axis([aa(1:2) roiA])
        hold on
        plot(y.chrgR(aPkInd),y.anglR(aPkInd),'rx')
        if aFwhmInd~=0
            plot(y.chrgR(aFwhmInd(1)),y.anglR(aFwhmInd(1)),'rx')
            plot(y.chrgR(aFwhmInd(2)),y.anglR(aFwhmInd(2)),'rx')
        end
        hold off
        xlabel('[pC/mrad]')
        ylabel('Angle [mrad]')
        set(gca,'fontsize',fntSz);

        %% save image
        fNameI = [pptDir,filesep,'Scan',scanNS,'_MgspcHttPpt_',sNmbS,'.png'];
        saveSameSize(fig1,'format','png','renderer','opengl','file',fNameI);

    end % shots loop

    %% waterfall plots
    set(0,'CurrentFigure',fig2)
    subplot(3,1,1)
    pcolor(x.mmt,[1:nmbSht],wtrFllM)
    shading interp
    colorbar
    title('Linear scale [pC/(MeV/c)]')
    xlabel('[MeV/c]')
    ylabel('shotnumber')
    aa = axis;
    axis([roiM aa(3:4)])
    set(gca,'fontsize',fntSz);

    subplot(3,1,2)
    pcolor(x.mmt,[1:nmbSht],log10(wtrFllM))
    colorbar
    shading interp
    xlabel('[MeV/c]')
    ylabel('shotnumber')
    title('Log scale')
    aa = axis;
    axis([roiM aa(3:4)])
    cc = caxis;
    if cc(2)>-3
        caxis([-3 cc(2)])
    end
    set(gca,'fontsize',fntSz);

    subplot(3,1,3)
    errorbar(x.mmt,mean(wtrFllM),std(wtrFllM))
    colorbar
    shading interp
    xlabel('[MeV/c]')
    ylabel('[pc/(MeV/c)]')

    aa = axis;
    axis([roiM aa(3:4)])
    cc = caxis;
    caxis([-3 cc(2)])
    set(gca,'fontsize',fntSz);


    set(0,'CurrentFigure',fig3)
    pcolor(y.angl,[1:nmbSht],wtrFllA)
    shading interp
    xlabel('[mrad]')
    ylabel('shotnumber')
    aa = axis;
    axis([roiA aa(3:4)])
    set(gca,'fontsize',fntSz);

    namOut = [anlNDir,filesep,'Scan',scanNS,'_mmtWaterFall.png'];
    saveSameSize(fig2,'format','png','renderer','opengl','file',namOut);

    namOut = [anlNDir,filesep,'Scan',scanNS,'_anglWaterFall.png'];
    saveSameSize(fig3,'format','png','renderer','opengl','file',namOut);

    %% save output for a scan
    namOut = [anlNDir,filesep,'Scan',scanNS,'_MSAnalysis.txt'];
    ok = fTxtOutV01(namOut,nmbDOutL,ttlP,dat);
    if addScan==1
        fAdd2LogFile(logPath,ttlP(3:end),dat(:,3:end))
    end
end % scan loop
