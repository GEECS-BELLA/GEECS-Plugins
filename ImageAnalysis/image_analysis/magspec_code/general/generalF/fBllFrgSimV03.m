function [Isig,T,yAx,peakPwr] = fBllFrgSimV03(cmpL,frgTW,lmb,alpD,dspOrd,sysDsp,figH,varargin)
%% BELLA Frog simulation
% [] = fBllFrgSimV01(cmpL,N,lmb0,alpD,dspOrd,sysDsp,figH,vargin)
%
% D: distabce between gratings [mm]
% frgTW: frog time window
% lmb0: central wavelength [nm]
% alpD: incident angle [deg]
% dspOrd: order of dispersion to be considered.
% sysDsp: system dispersions(7) (everything except compressor)
% figH: figure Handle
% vargin: eS, mSize for specified spectrum, or sWdt for gaussian

%% Written by Kei Nakamura
% 2014/3/5 ver.1: created
% 2014/7/22 ver.1b: accept single compL, output frog image
% 2014/11/26 ver.2: update
% 2015/3/31 ver.3: 800 nm
% 2015/7/24 ver.3.1: output tw/j

%% Used in
% bellaFrogPhaseMSV04
% bellaFrogFitTest

%% constants
lmb0 = lmb(1);
lmb1 = lmb(2);
lmb2 = lmb(3);

if nargin ==8 %gaussian, 1 var
    sWdt = varargin{1};
    mSize = 1024;       % matrix size for time and freq FFT
    pWdt = ((lmb0/800)^2)*940/sWdt;
else % specific, 3 inputs
    mSize = varargin{2};       % matrix size for time and freq FFT
    tStp = varargin{3};
end
[T,W,~] = fFrogFFTParaV02(mSize,tStp);   % get fourier parameter for frog analysis
%[T,tStp,W,omg0,~] = fFrogFFTParaV01(lmb0,mSize);

%% input dependent constants
[N,D] = fBellaCompClbV01(alpD,cmpL);
sysDsp(1:6-dspOrd) = 0;

%% calculation
if figH~=0
    set(0,'CurrentFigure',figH)
end

for jj=1:numel(cmpL)

    % dispersions from compressor
    cmpDsp = fCpaDispersionV02(D(jj),N,lmb0,alpD,-1,dspOrd); % compressor dispersions
    dsprsn = cmpDsp + sysDsp;   % total dispersions
    dsprsnF = fliplr(dsprsn./[-720,-120,-24,-6,-2,1,1]);   % for frog program format

    % e field
    if nargin == 8  % no specified spectrum
        etDiag = pulsegenerator(mSize,@fgaussian,pWdt,tStp,lmb0,[0 0 0 0],[],dsprsnF);
        etDiag = quickscale(etDiag);
    else % with specified input spectrum
        eS = varargin{1};
        sphase1 = polyval(fliplr(dsprsnF),W);
        esDiag = eS .* exp(1i * sphase1);
        etDiag = ifftc(esDiag);
    end

    % analysis
    peakPwr(jj) = round(100*fPkPowerAnl(etDiag,tStp))/100;   % peak power [TW/J]
    fwhmT = round(100*fwhm(IandP(etDiag),T))/100;   % pulse length in fwhm [fs]

    % information
    peakPTxt = [num2str(peakPwr(jj)) ' TW/J'];
    lengthTxt = [num2str(fwhmT) ' fs'];
    p2txt = ['p2= ',num2str(round(dsprsn(5)))];
    p3txt = ['p3= ',num2str(0.1*round(1e-2*dsprsn(4))),'k'];
    p4txt = ['p4= ',num2str(0.01*round(1e-4*dsprsn(3))),'M'];
    p5txt = ['p5= ',num2str(0.1*round(1e-5*dsprsn(2))),'M'];
    p6txt = ['p6= ',num2str(0.01*round(1e-7*dsprsn(1))),'G'];

    % simulate frog in theory
    if figH~=0
        subplot(3,3,jj)
        plotfrog(etDiag,etDiag,T,lmb0);
        colormap jet
        %axis([-round(frgTW/2),round(frgTW/2),390,425])
        axis([-round(frgTW/2),round(frgTW/2),380,420])
        aa = axis;
        hold on
        plot([aa(1) aa(2)],[0.5*lmb0 0.5*lmb0],'w--');
        plot([aa(1) aa(2)],[0.5*lmb1 0.5*lmb1],'w--');
        plot([aa(1) aa(2)],[0.5*lmb2 0.5*lmb2],'w--');
        text(-200,aa(4)-3,lengthTxt,'color',[1,1,1]);
        text(0,aa(4)-3,peakPTxt,'color',[1,1,1]);
        text(-200,aa(3)+10,p2txt,'color',[1,1,1]);
        text(-200,aa(3)+6,p3txt,'color',[1,1,1]);
        text(-200,aa(3)+2,p4txt,'color',[1,1,1]);
        text(0,aa(3)+6,p5txt,'color',[1,1,1]);
        text(0,aa(3)+2,p6txt,'color',[1,1,1]);
        xlabel('[fs]');
        ylabel('wavelength [nm]');
        %ttl = ['comp=',num2str(cmpL(jj)),' mm, /tau = ',num2str(round(fGetFwhmV04(T,IandP(etDiag)))),' fs'];
        ttl = ['comp=',num2str(cmpL(jj)),' mm'];
        title(ttl)
        hold off
    end

%     % simulate frog in exp
%     set(0,'CurrentFigure',fig6)
%     subplot(3,3,jj)
%     plotfrog(etDiag,etDiag,T,L0);
%     caxis([ccdTh/(ccdTh+ccdPk+1),1])
%     aa = axis;
%     axis([-round(frgTW/2),round(frgTW/2),1.06*aa(3),0.94*aa(4)])
%     xlabel('fs');
%     ylabel('nm');
%     ttl = ['comp=',num2str(cmpFrg),' mm, /tau = ',num2str(round(fGetFwhmV04(T,IandP(etDiag)))),' fs'];
%     title(ttl)
end

%% for output frog image (the last one)
if figH==0
[Isig,yAx] = fMakeIsig(etDiag,etDiag,T,lmb0);
end

% figure(10)
% plot(T,abs(etDiag.^2))
