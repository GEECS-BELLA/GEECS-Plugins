%% inport TERMITES data

%% Revisions
% 2018/11/29 v.1: create

%% initialization
clear all
%close all

fig1 = figure(1);
set(gca,'fontsize',14);
set(gcf,'color',[1 1 1]);
set(fig1, 'Position', [1200 850 900 650]);

fig2 = figure(2);
set(gca,'fontsize',14);
set(gcf,'color',[1 1 1]);
set(fig2, 'Position', [1650 700 900 650]);

fig3 = figure(3);
set(gca,'fontsize',14);
set(gcf,'color',[1 1 1]);
set(fig3, 'Position', [1200 0 900 650]);

fig4 = figure(4);
set(gca,'fontsize',14);
set(gcf,'color',[1 1 1]);
set(fig4, 'Position', [1650 0 900 650]);

%% input
szP = 512*1;    % size after padding

%% data
dir1 = '/Users/kee/Dropbox (Bella Center)/Documents/otherProjects/TERMITES/TERMITES data/';
fNam1 = [dir1,'TERMITES-RESULTS_2017-08-31-193026.hdf5'];   % max E, stretched
fNam2 = [dir1,'TERMITES-RESULTS_2017-08-31-204708.hdf5'];   % max E, compressed
fNam3 = [dir1,'TERMITES-RESULTS_2017-09-01-002353.hdf5'];   % low E, stretched
fNam4 = [dir1,'TERMITES-RESULTS_2017-09-01-013105.hdf5'];   % low E, compressed
fNam5 = [dir1,'TERMITES-RESULTS_2017-09-01-023941.hdf5'];   % low E, with PFT
fNam6 = [dir1,'INSIGHT-bella.hdf5'];   % insight
%h5disp(fNam1);

%% download field
[e1,xMm,yMm,lmbNm,omgFs] = fReadPhicoreH5(fNam6,0,1);
% bin E-field
%[e2,xMm2,yMm2,omgFs2] = fBin3DEfld(e1,xMm,yMm,omgFs,[2 2 1]);


% pad field
szE = size(e1);
[e2,xMm2,yMm2,omgFs2] = fPad3DEfld(e1,xMm,yMm,omgFs,[szP szP szE(3)]);

%% fig1,2: plot field in x domain
labels{1} = 'x [mm]';
labels{2} = 'y [mm]';
labels{3} = 'omg [rad/fs]';
fPlot3DEfld(e1,xMm,yMm,omgFs,labels,fig1)
fPlot3DEfld(e2,xMm2,yMm2,omgFs2,labels,fig2)

%% calculate in k
e1k = fftshift(fftshift(fft2(e1),1),2); % FFT to k-w domain, shifted
e1kB = fft2(e1); % FFT to k-w domain, not shifted
[~,kxMm1,~] = fFrogFFTParaV02(szE(2),xMm(2)-xMm(1)); %kx [rad/mm]
[~,kyMm1,~] = fFrogFFTParaV02(szE(1),yMm(2)-yMm(1)); %kx [rad/mm]

e2k = fftshift(fftshift(fft2(e2),1),2); % FFT to k-w domain, shifted
e2kB = fft2(e2); % FFT to k-w domain, not shifted
[~,kxMm2,~] = fFrogFFTParaV02(szP,xMm2(2)-xMm2(1)); %kx [rad/mm]
[~,kyMm2,~] = fFrogFFTParaV02(szP,yMm2(2)-yMm2(1)); %kx [rad/mm]

labels{1} = 'kx [rad/mm]';
labels{2} = 'ky [rad/mm]';
fPlot3DEfld(e1k,kxMm1,kyMm1,omgFs,labels,fig3,[-0.2,0.2,-0.2,0.2,2.2 2.4])
fPlot3DEfld(e2k,kxMm2,kyMm2,omgFs2,labels,fig4,[-0.2,0.2,-0.2,0.2,2.2 2.4])
