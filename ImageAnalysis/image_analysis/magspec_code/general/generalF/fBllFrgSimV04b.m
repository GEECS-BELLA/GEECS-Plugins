function [Isig,T,yAx,peakPwr] = fBllFrgSimV04b(cmpL,frgTW,lmb,alpD,dspOrd,sysDsp,figH,varargin)
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
% 2015/9/29 ver.4: 12th order
% 2017/3/30 ver.4: 1x7 array

%% Used in
% bellaFrogFirstAnlV02

%% constants
lmb0 = lmb(1);
lmb1 = lmb(2);
lmb2 = lmb(3);
fctArry = zeros(13,1);
fctArry(1) = 1;
for ii=1:12
    fctArry(ii+1)= factorial(ii);
end
fctArry = -flipud(fctArry);

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
sysDsp(1:12-dspOrd) = 0;

%% calculation
if figH~=0
    set(0,'CurrentFigure',figH)
end

for jj=1:numel(cmpL)

    % dispersions from compressor
    cmpDsp = fCpaDispersionV03(D(jj),N,lmb0,alpD,-1,dspOrd); % compressor dispersions
    dsprsn = cmpDsp + sysDsp;   % total dispersions
    dsprsnF = dsprsn./fctArry';   % for frog program format
    %dsprsnF = dsprsnF(13-dspOrd:end);   % take only needed

    % e field
    if nargin == 8  % no specified spectrum
        etDiag = pulsegenerator(mSize,@fgaussian,pWdt,tStp,lmb0,[0 0 0 0],[],dsprsnF);
        etDiag = quickscale(etDiag);
    else % with specified input spectrum
        eS = varargin{1};
        sphase1 = polyval(dsprsnF,W);
        esDiag = eS .* exp(1i * sphase1);
        etDiag = ifftc(esDiag);
    end

    % analysis
    peakPwr(jj) = round(100*fPkPowerAnl(etDiag,tStp))/100;   % peak power [TW/J]
    fwhmT = round(100*fwhm(IandP(etDiag),T))/100;   % pulse length in fwhm [fs]

    % information
    peakPTxt = [num2str(peakPwr(jj)) ' TW/J'];
    lengthTxt = [num2str(fwhmT) ' fs'];
    p2txt = ['p2= ',num2str(round(dsprsn(11)))];
    p3txt = ['p3= ',num2str(0.1*round(1e-2*dsprsn(10))),'k'];
    p4txt = ['p4= ',num2str(0.01*round(1e-4*dsprsn(9))),'M'];
    p5txt = ['p5= ',num2str(0.1*round(1e-5*dsprsn(8))),'M'];
    p6txt = ['p6= ',num2str(0.01*round(1e-7*dsprsn(7))),'G'];

    % simulate frog in theory
    if figH~=0
        subplot(3,7,jj)
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

        % spectral phase plot
        subplot(3,7,jj+7)
        [~,~,~,xPlt,yPlt] = plotcmplxV2(W(230:512-230),esDiag(230:512-230)); % get phase out
        xlabel('\omega - \omega_0 [rad/fs]')

        % GVD plot
        dphi = diff(yPlt(:,2))/(xPlt(2)-xPlt(1));   % dphi/dw [rad/(rad/fs)]
        dphi = [dphi;dphi(end)];
        pFit = polyfit(xPlt,dphi,1);


        subplot(3,7,jj+14)
        plot(xPlt,dphi)
        hold on
        plot(xPlt,polyval(pFit,xPlt))
        hold off
        aa = axis;
        axis([aa(1:2) -50 50])
        title(['GVD=' num2str(pFit(1),4)])
        xlabel('\omega - \omega_0 [rad/fs]')
        ylabel('d\phi/d\omega')
        %plotcmplxV2(T(230:512-230),etDiag(230:512-230))
    end
end

%% for output frog image (the last one)
if figH==0
[Isig,yAx] = fMakeIsig(etDiag,etDiag,T,lmb0);
end

% figure(10)
% plot(T,abs(etDiag.^2))
