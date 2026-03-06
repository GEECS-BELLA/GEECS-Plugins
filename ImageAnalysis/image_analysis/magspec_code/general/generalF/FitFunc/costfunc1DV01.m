function z = costfunc1DV01(popu,X,Y,tfct,argf)
% Cost functions used by the 'gcs' algorithm
%
% z = costfunc(popu,X,Y,tfct,argf)
%
% popu: matrix for which each column contains parameters of costfunc
%       each column is a cell, the matrix is the population of cells
% X:    x
% Y:    f(x) to be fitted
% tfct: type of costfunc, 1: Gaussian, 2: linear, 3: polynomial 4:
% exponential, 5: gate, 6: one root polynomial, 7: enge function
% argf: optional parameter if needed. In polynomial case, it is the order
%       of the polynom
%
% z:    cost function, sum of mean square (can be changed if needed)
%

%% cost function for gcs
% Kei Nakamura
% succeeded from costfunc from
% Guillaume Plateau Feb. 23rd 2010
%            grplateau@lbl.gov
%            yguig2@yahoo.fr
% Created on Thursday, January 6th 2011.
% Modified on Thursday,

%% Example:
% Fitting function is g(x) = p1*x^2 + p2*log(x) +p3
%
% popu will be filled of columns of [p1;p2;p3] to be tested
%
% Under "Otherwise", you define g(x) ("longfct")
% over the matrix popu ("longV"), and the axis X ("longX"):
%
% longfct=longV(1,:).*(longX.^2) + longV(2,:).*log(longX) + longV(3,:);
% --------------------------------------------------------------------
%
% popu is the trial parameters
% X is X axis, Y is the answer (to be compared with fit results)
% longfct is the connected answer to be compared

%% main body
if ~isempty(popu)

    longV = matrix((ones(size(X,2),1)*matrix(popu',1,-1)),size(X,2)*...
        size(popu,2),-1)';
    longX = matrix(X'*ones(1,size(popu,2)),1,-1);
    longY = matrix(Y'*ones(1,size(popu,2)),1,-1);

    switch tfct
        case 1
            %% Gaussian
            longfct=longV(1,:)+longV(2,:).*exp((-1/2)*...
                (((longX-longV(3,:))./longV(4,:)).^2));

        case 2
            %% Linear
            longfct=longV(1,:)+longV(2,:).*longX;

        case 3
            %% Polynomial
            longfct=longV(1,:);
            for i=1:argf
                longfct=longfct+longV(i+1,:).*(longX.^i);
            end

        case 4
            %% Exponential
            longfct=longV(1,:)+longV(2,:).*exp(-longV(3,:).*...
                (longX-longV(4,:)));

        case 5
            %% Gate
            longfct=((longX>=longV(1,:))&(longX<longV(2,:))).*longV(3,:);

        case 6
            %% One root polynomial
            longfct=longV(1,:)+longV(2,:).*...
                ((longX-longV(3,:)).^longV(4,:));

        case 7
            %% enge function 1st, argf(1) = gap, argf(2) = order
            expnt = longV(1,:);     % inside of exp
            for ii=1:argf(2)
                expnt = expnt + longV(ii+1,:).*(longX/argf(1)).^ii;
            end
            longfct = 1./(1+exp(expnt));

        case 8
            %% for frog
            %tic
            %nmbImg = numel(longX)/size1D; % number of trial
            cmpLPeak = argf{1};
            [~,nmbImg] = size(cmpLPeak);    % number of image, for fitting
            cmrPara = argf{3};
            size1D = numel(cmrPara.T)*numel(cmrPara.W)*nmbImg;   % 1d size
            nmbTry = numel(longX)/size1D; % number of trial
            [nmbPrm,~] = size(longV);       % number of parameter
            longV = longV(:,1:size1D:end);  % deminish fit parameter
            fitPrm = longV';    % nmbTry x fitP array
            fitPrm = [zeros(nmbTry,5-nmbPrm),fitPrm,zeros(nmbTry,2)];  % dispersions
            longfct(1,:)= fBllFrgSim4FitV04(argf{1},fitPrm,argf{2},argf{3},argf{5},0,argf{6},argf{7});
            %toc
         case 9
            %% for frog, with w-offset
            %tic
            %nmbImg = numel(longX)/size1D; % number of trial
            cmpLPeak = argf{1};
            [~,nmbImg] = size(cmpLPeak);    % number of image, for fitting
            cmrPara = argf{3};
            size1D = numel(cmrPara.T)*numel(cmrPara.W)*nmbImg;   % 1d size
            nmbTry = numel(longX)/size1D; % number of trial
            [nmbPrm,~] = size(longV);       % number of parameter
            longV = longV(:,1:size1D:end);  % deminish fit parameter
            fitPrm = longV';    % nmbTry x fitP array
            fitPrm = [zeros(nmbTry,6-nmbPrm),fitPrm,zeros(nmbTry,2)];  % dispersions
            longfct(1,:)= fBllFrgSim4FitV05(argf{1},fitPrm,argf{2},argf{3},argf{5},0,argf{6},argf{7});
            %toc
         case 10
            %% for frog 12th
            cmpLPeak = argf{1};
            [~,nmbImg] = size(cmpLPeak);    % number of image, for fitting
            cmrPara = argf{3};
            size1D = numel(cmrPara.T)*numel(cmrPara.W)*nmbImg;   % 1d size
            nmbTry = numel(longX)/size1D; % number of trial
            [nmbPrm,~] = size(longV);       % number of parameter
            longV = longV(:,1:size1D:end);  % deminish fit parameter
            fitPrm = longV';    % nmbTry x fitP array
            fitPrm = [zeros(nmbTry,11-nmbPrm),fitPrm,zeros(nmbTry,2)];  % dispersions
            longfct(1,:)= fBllFrgSim4FitV05(argf{1},fitPrm,argf{2},argf{3},argf{5},0,argf{6},argf{7});
        otherwise
            %% special fit (define your own function here)
            longfct=0;
    end

    z = sum(matrix((longY-longfct).^2,size(Y,2),-1),1);
else
	z=[];
end
