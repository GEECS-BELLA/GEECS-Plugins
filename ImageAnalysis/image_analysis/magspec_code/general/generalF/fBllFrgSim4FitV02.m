function [arry,pulse] = fBllFrgSim4FitV02(cmpLPeak,dsprsnIn,cmpPara,cmrPara,axesP,diag,figH,varargin)
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
% 2014/7/31 ver.2: #ofTry
% 2014/8/4 ver.2b: # of image

%% Used in
% bellaFrogFitTest

%% inputs
cmpL = cmpLPeak(1,:);   % compressor locations
peakCnt = cmpLPeak(2,:);    % peak counts for frog image
lmb0 = cmpPara.lmb0;    % lambda 0, central wavelength
alpD = cmpPara.alpD;    % incident angle [deg]
%[N,D] = fBellaCompClbV01(alpD,cmpL);
dspOrd = cmpPara.dspOrd;    % dispersion order
cmrThre = cmrPara.cmrThre;  % camera threshold count
anlThre = cmrPara.anlThre;  % analysis threshold
cmrSig = cmrPara.cmrSig;    % camera sigma
binNmb = cmrPara.binNmb;    % bin #
wvlRoi = cmrPara.wvlRoi;
frogWvl = cmrPara.frogWvl;  % hardware setting

%% input for spectrum
if numel(varargin{1})==1   % Gaussian spectrum
    sWdt = varargin{1};
    pWdt = ((lmb0/800)^2)*940/sWdt;
end

%% constants
mSize = 1024;       % matrix size for time and freq FFT
[T,tStp,W,omg0] = fFrogFFTParaV01(lmb0,mSize);
[rawFrgT,rawFrgL,~] = fFrogClbV01(diag,frogWvl,[wvlRoi(1:2),numel(axesP.nm)*2^binNmb]);

%% input dependent constants
diagDsp = fBllDiagDsprsnV01(lmb0,diag,dspOrd);
%[xAxM,yAxM] = meshgrid(axesP.fs,axesP.nm);
[xAxM,yAxM] = meshgrid(rawFrgT(1:0.5*numel(rawFrgT)),rawFrgL); % mesh for 2d interp, half in time
[nmbTry,~] = size(dsprsnIn);    % number of try, for fitting
[~,nmbImg] = size(cmpLPeak);    % number of image, for fitting

%% allocation
etDiag = zeros(1,mSize); % e-field, spec, intaraction point
esDiag = zeros(1,mSize); % e-field, spec, intaraction point
IsigP = zeros(size(xAxM));  % Intensity signal (camera resolution)
tSz = numel(axesP.fs);  % time size
IsigPB = zeros([numel(axesP.nm),tSz*nmbImg]);    % binned Isig
size1D = numel(IsigPB*nmbImg); % size for 1D output
arry = zeros(size1D*nmbTry,1);   % output 1D array

%% calculation
for jj=1:nmbTry

    for ii=1:nmbImg

        %% dispersions calculation
        [N,D] = fBellaCompClbV01(alpD,cmpL(ii));
        cmpDsp = fCpaDispersionV02(D,N,lmb0,alpD,-1,dspOrd); % compressor dispersions
        dsprsn = cmpDsp + dsprsnIn(jj,:) + diagDsp;   % total dispersions
%         disp(cmpDsp(5)')
%         disp(dsprsnIn(jj,5)')
%         disp(diagDsp(5)')
%         disp(dsprsn(5)')
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
        mxPower = 0;   % max power [TW/J]
        [Isig,yAx,T2] = fMakeIsig(etDiag,etDiag,T,lmb0,[wvlRoi(1)-1,wvlRoi(2)+1],[rawFrgT(1)-10,-(rawFrgT(1)-10)]);   % get theoretical image
        IsigP(:,:) = interp2(T2,yAx,Isig,xAxM,yAxM);   % 2d interplate to raw frog image size
        IsigP(:,:) = (cmrThre+peakCnt(ii))*IsigP/max(max(IsigP));  % normalized to ccd count+thre
        IsigP(:,:) = cmrSig*randn(size(IsigP)) + IsigP;      % add rms noise, image for process
        IsigP(:,:) = fImgSmooth9(IsigP);                % 2D smoothing
        id1 = tSz*(ii-1)+1; id2 = tSz*ii;% index for convenience
        [~,~,IsigPB(:,id1:id2)] = fFrogFitBinRoiV01(rawFrgT,rawFrgL,IsigP,binNmb);   % bin and roi
        IsigPB(:,id1:id2) = IsigPB(:,id1:id2)-cmrThre; % subtract camera threshold
        IsigPB(:,id1:id2) = IsigPB(:,id1:id2)*peakCnt(ii)/max(max(IsigPB(:,id1:id2)));  % re-normalized to peak


        %% figure for multiple comp location
        if figH~=0&&nmbImg>1
            set(0,'CurrentFigure',figH)

            %% intensity profile
            subplot(6,nmbImg,4*nmbImg+ii)
            etDiag(:) = fTWpJ(etDiag,tStp); % normalize to TW/J for intensity
%             itDiag = abs(etDiag).^2;
%             engInt = sum(itDiag);   % energy, time-integrated intensity
%             etDiag(:) = etDiag*sqrt(1000/(engInt*tStp));   % normalization
            [~,ind1] = min(abs(T-rawFrgT(1)));   % index 1
            [~,ind2] = min(abs(T-rawFrgT(end)));   % index 1
            [~,~,~,~,yPlt] = plotcmplxV2(T(ind1:ind2),etDiag(ind1:ind2),[],[],[]);
            mxPower = max(yPlt(:,1));   % max power [TW/J]
            title([num2str(round(fs)),' [fs], ' num2str(round(mxPower)),' [TW/J]'])
            xlabel('[fs]')

            %% intensity profile in log
            subplot(6,nmbImg,5*nmbImg+ii)
            plotcmplxV2log(T,etDiag,[],[],[]);
            title(['comp=' num2str(cmpL(ii))])
            xlabel('[fs]')

            %% information
            p2txt = ['p2= ',num2str(round(dsprsn(5)))];
            p3txt = ['p3= ',num2str(0.1*round(1e-2*dsprsn(4))),'k'];
            p4txt = ['p4= ',num2str(0.01*round(1e-4*dsprsn(3))),'M'];
            p5txt = ['p5= ',num2str(0.1*round(1e-5*dsprsn(2))),'M'];
            p6txt = ['p6= ',num2str(0.01*round(1e-7*dsprsn(1))),'G'];

            hold on
            text(T(1),-1,p2txt,'color',[0,0,0]);
            text(T(1),-2,p3txt,'color',[0,0,0]);
            text(T(1),-3,p4txt,'color',[0,0,0]);
            text(abs(T(1)/2),-1,p5txt,'color',[0,0,0]);
            text(abs(T(1)/2),-2,p6txt,'color',[0,0,0]);
%             text(rawFrgT(1)+10,0.5*wvlRoi(1)+4,p3txt,'color',[1,1,1]);
%             text(rawFrgT(1)+10,0.5*wvlRoi(1)+1,p4txt,'color',[1,1,1]);
%             text(0,0.5*wvlRoi(1)+4,p5txt,'color',[1,1,1]);
%             text(0,0.5*wvlRoi(1)+1,p6txt,'color',[1,1,1]);
            hold off

        end

        IsigPB(:,:) = IsigPB.*(IsigPB>anlThre);    % thresholding for analysis

        %% output
        strt = (jj-1)*size1D+1; % start index
        fnsh = strt+size1D-1; % end index
        arry(strt:fnsh,1) = reshape(IsigPB,numel(IsigPB),1);
    end

end

%% figure
if figH~=0&&nmbImg==1
    set(0,'CurrentFigure',figH)

    %% intensity profile
    itDiag = abs(etDiag).^2;
    engInt = sum(itDiag);   % energy, time-integrated intensity
    etDiag(:) = etDiag*sqrt(1000/(engInt*tStp));   % normalization
    [~,ind1] = min(abs(T-rawFrgT(1)));   % index 1
    [~,ind2] = min(abs(T-rawFrgT(end)));   % index 1
    subplot(3,3,3)
    [~,~,~,~,yPlt] = plotcmplxV2(T(ind1:ind2),etDiag(ind1:ind2),[],[],[]);
    mxPower = max(yPlt(:,1));   % max power [TW/J]
    %plotcmplxV2(T,etDiag,[],[],[]);
    title('blue: power, green: phase')
    xlabel('[fs]')

    %% intensity profile in log
    subplot(3,3,2)
    %plotcmplxV2log(T(ind1:ind2),etDiag(ind1:ind2),[],[],[]);
    plotcmplxV2log(T,etDiag,[],[],[]);
    title('log scale power')
    xlabel('[fs]')

    %% information
    p2txt = ['p2= ',num2str(round(dsprsn(5)))];
    p3txt = ['p3= ',num2str(0.1*round(1e-2*dsprsn(4))),'k'];
    p4txt = ['p4= ',num2str(0.01*round(1e-4*dsprsn(3))),'M'];
    p5txt = ['p5= ',num2str(0.1*round(1e-5*dsprsn(2))),'M'];
    p6txt = ['p6= ',num2str(0.01*round(1e-7*dsprsn(1))),'G'];

    %% image for fitting, log scale
    subplot(3,3,7)
    pcolor(axesP.fs,axesP.nm,log10(IsigPB))
    shading interp
    aa = caxis;
    caxis([-2,aa(2)])
    colorbar
    axis([rawFrgT(1),0,0.5*wvlRoi])
    xlabel('time [fs]')
    ylabel('wavelength [nm]')
    title('fitted image, log')

    %% fitting result, log scale
    subplot(3,3,8)
    pcolor(T2,yAx,log10(peakCnt*Isig/max(max(Isig))))
    shading interp
    aa = caxis;
    caxis([-2,aa(2)])
    colorbar
    axis([rawFrgT(1),rawFrgT(end),0.5*wvlRoi])
    xlabel('time [fs]')
    ylabel('wavelength [nm]')
    title('fitting result, log')

    %% fitting result, linear scale
    subplot(3,3,9)
    pcolor(T2,yAx,peakCnt*Isig/max(max(Isig)))
    shading interp
    axis([rawFrgT(1),rawFrgT(end),0.5*wvlRoi])
    colorbar
    hold on
    text(rawFrgT(1)+10,0.5*wvlRoi(1)+7,p2txt,'color',[1,1,1]);
    text(rawFrgT(1)+10,0.5*wvlRoi(1)+4,p3txt,'color',[1,1,1]);
    text(rawFrgT(1)+10,0.5*wvlRoi(1)+1,p4txt,'color',[1,1,1]);
    text(0,0.5*wvlRoi(1)+4,p5txt,'color',[1,1,1]);
    text(0,0.5*wvlRoi(1)+1,p6txt,'color',[1,1,1]);
    xlabel('fs');
    ylabel('nm');
    %ttl = ['comp=',num2str(cmpL),' mm, /tau = ',num2str(round(fs)),' fs'];
    %title(ttl)
    title('fitting result, linear')
    hold off

elseif figH~=0&&nmbImg>=1
    set(0,'CurrentFigure',figH)

    % linear scale
    subplot(6,nmbImg,1+nmbImg:2*nmbImg)
    pcolor(IsigPB)
    shading interp
    colorbar
    [szY,szX] = size(IsigPB);
    szXD = szX/nmbImg;  % size x per image
    hold on
    plot([0 szX],[0.5*szY 0.5*szY],'w--')
    for pp=1:nmbImg
        plot([szXD*pp-szXD/5 szXD*pp-szXD/5],[0 szY],'w--')
    end
    hold off
    tx1 = ['p2=' num2str(1e-3*round(dsprsnIn(1,5))) ' kfs^2'];
    tx2 = [', p3=' num2str(1e-1*round(1e-2*dsprsnIn(1,4))) ' kfs^3'];
    tx3 = [', p4=' num2str(1e-3*round(1e-3*dsprsnIn(1,3))) ' Mfs^4'];
    tx4 = [', p5=' num2str(1e-2*round(1e-4*dsprsnIn(1,2))) ' Mfs^5'];
    tx5 = [', p6=' num2str(round(1e-6*dsprsnIn(1,1))) ' Mfs^6'];
    title([tx1 tx2 tx3 tx4 tx5])

    % log scale
    subplot(6,nmbImg,1+nmbImg*3:4*nmbImg)
    pcolor(log10(IsigPB))
    shading interp
    colorbar
    hold on
    plot([0 szX],[0.5*szY 0.5*szY],'w--')
    for pp=1:nmbImg
        plot([szXD*pp-szXD/5 szXD*pp-szXD/5],[0 szY],'w--')
    end
    hold off
    tx1 = ['p2=' num2str(1e-3*round(dsprsnIn(1,5))) ' kfs^2'];
    tx2 = [', p3=' num2str(1e-1*round(1e-2*dsprsnIn(1,4))) ' kfs^3'];
    tx3 = [', p4=' num2str(1e-3*round(1e-3*dsprsnIn(1,3))) ' Mfs^4'];
    tx4 = [', p5=' num2str(1e-2*round(1e-4*dsprsnIn(1,2))) ' Mfs^5'];
    tx5 = [', p6=' num2str(round(1e-6*dsprsnIn(1,1))) ' Mfs^6'];
    title([tx1 tx2 tx3 tx4 tx5])
end

pulse.fs = fs;
pulse.mxPower = mxPower;
pulse.dsprsn = dsprsn;
