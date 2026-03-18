function arry = fBllFrgSim4FitV01b(cmpLPeak,dsprsn,cmpPara,cmrPara,axesP,diag,figH,varargin)
%% BELLA Frog simulation for fitting
% arry = fBllFrgSim4FitV01(cmpL,dsprsn,cmpPara,peakC,cmrPara,varargin)
% calculats simulated frog image for fitting
%
% cmpLPeak: 2 x N array of compressor location and peak counts
% dsprsn: array for dispersions(7), at compressor input
% cmpPara: compressor parameter structure
%           .N: groove density [nmb/mm]
%           .alpD: incident angle [deg]
%           .lmb0: central wavelength [nm]
%           .dspOrd: order of dispersion to be considered.
% cmrPara: forg camera parameter
%           .cmrStd: sigma for camera for noise level
%           .cmrThre: threshold for camera
%           .anlThre: threshold for analysis
%           .cmrX: camera x axis [fs]
%           .cmrY: camera y axis [nm]
%           .binNmb: bin#
%           .wvlRoi: wavelength roi
% axesP: axis.fs and axis.nm
% diag: diagnostic name, FrogHPD' or 'Frog_LB'
% figL: figure logic
% vargin: eS for specified spectrum, or sWdt for gaussian

%% Written by Kei Nakamura
% 2014/7/23 ver.1: modified fBllFrgSim
% 2014/7/31 ver.1b: modified sig process, allocation

%% Used in
% bellaFrogFitTest

%% inputs
cmpL = cmpLPeak(1,:);   % compressor locations
peakCnt = cmpLPeak(2,:);    % peak counts for frog image
lmb0 = cmpPara.lmb0;    % lambda 0, central wavelength
alpD = cmpPara.alpD;    % incident angle [deg]
N = cmpPara.N;          % groove density [/mm]
dspOrd = cmpPara.dspOrd;    % dispersion order
cmrThre = cmrPara.cmrThre;  % camera threshold count
anlThre = cmrPara.anlThre;  % analysis threshold
cmrSig = cmrPara.cmrSig;    % camera sigma
binNmb = cmrPara.binNmb;    % bin #
wvlRoi = cmrPara.wvlRoi;

%% input for spectrum
if numel(varargin{1})==1   % Gaussian spectrum
    sWdt = varargin{1};
    pWdt = ((lmb0/800)^2)*940/sWdt;
end

%% constants
biasL = 1043.345;   % offset for L from stage encoder [mm]
btaCmposD = 17.233;    % beta comp offset [deg]
mSize = 1024;       % matrix size for time and freq FFT
tWnd = 2000;    % time window (whole, fs), default 2000, HR 5000
tStp = tWnd/mSize; % time step
T = (-mSize/2 : mSize/2 - 1) * tStp;    % time
deltaomega = 2*pi/(T(end) - T(1));  % d omega
omg0 = ltow(lmb0);  % peak omega
W = (-mSize/2:mSize/2-1)*deltaomega + omg0;   % omega axis for DFT
[rawFrgT,rawFrgL,~] = fFrogClbV01(diag,800,[wvlRoi(1:2),numel(axesP.nm)*2^binNmb]);

%% input dependent constants
alpR = alpD*pi/180;   % alpha in radian
btaCmposR = btaCmposD*pi/180;    % beta comp offset [rad]
bta0R = alpR - btaCmposR;   % beta0, designed beta [rad]
D = (cmpL + biasL)*cos(bta0R); % D, grating distance [mm]
dsprsn(1:6-dspOrd) = 0; % 0 dispersions for not considered order
diagDsp = fBllDiagDsprsnV01(lmb0,diag,dspOrd);
%[xAxM,yAxM] = meshgrid(axesP.fs,axesP.nm);
[xAxM,yAxM] = meshgrid(rawFrgT(1:0.5*numel(rawFrgT)),rawFrgL); % mesh for 2d interp, half in time

%% allocation
etDiag = zeros(1,mSize); % e-field, spec, intaraction point
esDiag = zeros(1,mSize); % e-field, spec, intaraction point
IsigP = zeros(size(xAxM));  % Intensity signal (camera resolution)
IsigPB = zeros([numel(axesP.nm),numel(axesP.fs)]);    % binned Isig

%% calculation
for jj=1:numel(cmpL)

    %% dispersions calculation
    cmpDsp = fCpaDispersionV02(D(jj),N,lmb0,alpD,-1,dspOrd); % compressor dispersions
    dsprsn = cmpDsp + dsprsn + diagDsp;   % total dispersions
    dsprsnF = fliplr(dsprsn./[-720,-120,-24,-6,-2,1,1]);   % for frog program format

    %% e field calculation
    if numel(varargin{1})==1  % no specified spectrum
        etDiag(:,:) = pulsegenerator(mSize,@fgaussian,pWdt,tStp,lmb0,[0 0 0 0],[],dsprsnF);
        etDiag(:,:) = quickscale(etDiag);
    else % with specified input spectrum
        eS = varargin{1};
        sphase1 = polyval(fliplr(dsprsnF),W - omg0);
        esDiag(:,:) = eS .* exp(1i * sphase1);
        etDiag(:,:) = ifftc(esDiag);
    end

    %% simulate frog in theory
    fs = fGetFwhmV04(T,IandP(etDiag));  % pulse length
    [Isig,yAx,T] = fMakeIsig(etDiag,etDiag,T,lmb0,[wvlRoi(1)-1,wvlRoi(2)+1],[rawFrgT(1)-10,-(rawFrgT(1)-10)]);   % get theoretical image
    IsigP(:,:) = interp2(T,yAx,Isig,xAxM,yAxM);   % 2d interplate to raw frog image size
    IsigP(:,:) = (cmrThre+peakCnt(jj))*IsigP/max(max(IsigP));  % normalized to ccd count+thre
    IsigP(:,:) = cmrSig*randn(size(IsigP)) + IsigP;      % add rms noise, image for process
    IsigP(:,:) = fImgSmooth9(IsigP);                % 2D smoothing
    [~,~,IsigPB(:,:)] = fFrogFitBinRoiV01(rawFrgT,rawFrgL,IsigP,binNmb);   % bin and roi
    IsigPB(:,:) = IsigPB-cmrThre; % subtract camera threshold
    IsigPB(:,:) = IsigPB.*(IsigPB>anlThre);    % thresholding for analysis

    if figH~=0
        set(0,'CurrentFigure',figH)
        %% information
        p2txt = ['p2= ',num2str(round(dsprsn(5)))];
        p3txt = ['p3= ',num2str(0.1*round(1e-2*dsprsn(4))),'k'];
        p4txt = ['p4= ',num2str(0.01*round(1e-4*dsprsn(3))),'M'];
        p5txt = ['p5= ',num2str(0.1*round(1e-5*dsprsn(2))),'M'];
        p6txt = ['p6= ',num2str(0.01*round(1e-7*dsprsn(1))),'G'];

        %% plot for test
        subplot(2,2,1)
        pcolor(T,yAx,peakCnt(jj)*Isig/max(max(Isig)))
        shading interp
        %plotfrog(etDiag,etDiag,T,lmb0);
        axis([rawFrgT(1),rawFrgT(end),0.5*wvlRoi])
        colorbar
        hold on
        text(rawFrgT(1),0.5*wvlRoi(1)+7,p2txt,'color',[1,1,1]);
        text(rawFrgT(1),0.5*wvlRoi(1)+4,p3txt,'color',[1,1,1]);
        text(rawFrgT(1),0.5*wvlRoi(1)+1,p4txt,'color',[1,1,1]);
        text(0,0.5*wvlRoi(1)+4,p5txt,'color',[1,1,1]);
        text(0,0.5*wvlRoi(1)+1,p6txt,'color',[1,1,1]);
        xlabel('fs');
        ylabel('nm');
        ttl = ['comp=',num2str(cmpL(jj)),' mm, /tau = ',num2str(round(fs)),' fs'];
        title(ttl)
        hold off

        subplot(2,2,3)
        pcolor(T,yAx,log10(peakCnt(jj)*Isig/max(max(Isig))))
        shading interp
        aa = caxis;
        caxis([-2,aa(2)])
        colorbar
        aa = axis;
        axis([-round(513.8637/2),0,0.5*wvlRoi])
        xlabel('time [fs]')
        ylabel('wavelength [nm]')

        subplot(2,2,2)
        pcolor(axesP.fs,axesP.nm,log10(IsigPB))
        shading interp
        aa = caxis;
        caxis([-2,aa(2)])
        colorbar
        aa = axis;
        axis([-round(513.8637/2),0,aa(3:4)])
        xlabel('time [fs]')
        ylabel('wavelength [nm]')

        subplot(2,2,4)
        plot(reshape(IsigPB,numel(IsigPB),1),'b-')
    end

end

arry = reshape(IsigPB,numel(IsigPB),1);
