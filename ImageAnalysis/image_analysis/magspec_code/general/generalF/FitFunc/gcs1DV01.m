function [best,cost,nItr,dur,evl]=gcs1DV01(X,Y,tfct,argf,N,be,ro,d,nvar,span,init,tol,itmax,durmax,varargin)
% gcs MINIMIZES the function costfunc1DV01(x,y,...)
% [best,cost,nItr,dur,evl]=gcs1DV01(X,Y,tfct,argf,N,be,ro,d,nvar,span,init,tol,itmax,durmax)
%
% X,Y: parameters to be fit
% tfct: type of costfunc, see help costfunc1DV01
% N: number of cells
% be: multiplicative factor (be > 1/N)
% ro: mutation factor
% d: number of new cells at each iteration (0 <= d < N)
% nvar: optimized space dimension
% span: matrix [nvar 2] defining the range for each variable
% init: optional (nvar,1) initial values... if [], initialization by randomly distributed values
% tol;	     % threshold, if the cost goes below the algorithm stops
% itmax;      % maximum number of iterations
% durmax;     % maximum duration of calculation
% Mode: 'global' -> look for the global optimum
%
% dur;        % time duration of the optimization
% best;       % best cell
% cost;       % cost of the best cell
% nItr;     % number of iterations used

%% Clonal Selection for 1d ver.1
%
% Kei Nakamura
% 2010/2/23 original from Guillaume Plateau gcs.m
% 2011/1/6 ver.1: Created
% Modified on
%
% Algorithm: Leanardo N. de Castro, J. Von Zuben
% "Learning and Optimization Using the Clonal Selection Principle"
% IEEE Transactions on Evolutionary Computation Vol. 6 No. 3, Juin 2002,...
%     pages 239-251

%% Default parameters
L = span(:,2)-span(:,1);    % width of the region of interest (roi)
if (N<=0)||isempty(N)       % in case N is not specified
    N=(round(max(L)/2)>50)*50+(round(max(L)/2)<=50)*round(max(L)/2);
end
if (be<=0)||isempty(be)     % in case be is not specified
    be=15/N;
end
if (ro<-1)||isempty(ro)     % in case ro is not specified
    ro=10;
end
if (d<0)||isempty(d)        % in case d is not specified
    d=round(N/10);
end
tic;    % Start the stopwatch

%% Initial population
[cand_popu,span,L,C] = cs_init(N,nvar,span); % C:left corner of the roi
%size(init,1)
if isempty(init)    % if init is not specified
	popu=cand_popu;
    disp('no init')
elseif (~isempty(init))&&(size(init,1)==nvar)&&(size(init,2)==N) % if init is specified
	popu=init;
    %disp('one init')
elseif (~isempty(init))&&(size(init,1)==nvar)&&(size(init,2)==1) % if multiple init is specified
    popu=init*ones(1,N);
    %disp('>one init')
else
	popu=cand_popu; % if init is ill-specified
    disp('bad init')
end

imag_popu = costfunc1DV01(popu,X,Y,tfct,argf);
evl=imag_popu(1,1);
%disp(['initial cost' num2str(round(1e-4*evl))]);
%disp(round(popu(:,1)))
k=1;
nItr=1;

%% Mutation rate
taux_mut_clones = clonage((exp(-ro*linspace(1,0,N))'*ones(1,nvar))',be,N);
% vecteur ligne de longueur N duplique ennvar lignes puis clone

%% Main iterations
while continuer_gCS(evl,k,toc(),tol,itmax,durmax)
	% Sort -> Clone -> Mutation -> Insertion
    %tic
	[popu,imag_popu] = trier(popu,imag_popu);
	[popu,imag_popu] = inserer(gcs_mutation(clonage(popu,be,N),...
        nvar,be,N,taux_mut_clones,span,C),popu,imag_popu,N,...
        X,Y,tfct,argf);

	% Injection of the randomly distributed new values
	[popu,imag_popu] = forcer((L*ones(1,d)).*rand(nvar,d)+...
        (C*ones(1,d)),popu,imag_popu,N,d,X,Y,tfct,argf);

	% Storage of the best cell's image
	evl(1,k) = imag_popu(1,1);

    % for frog fit
    %disp(['itr=',num2str(k),', cost=',num2str(round(1e-4*evl(1,k)))])
    %disp(round(popu(:,1)))

    % plot
    %if ~isempty(varargin{1})  % Gaussian spectrum
    if ~isempty(varargin)  % Gaussian spectrum
        set(0,'CurrentFigure',varargin{1})
        costPlot(k) = evl(1,k);
        plot(1:k,costPlot,'bo')
        drawnow
    end

	nItr=k;
	k=k+1;
    %toc
end

best = popu(:,1);
cost = evl(1,nItr);
dur=toc;

end

% Called functions

%% cs_init
function [popu,span,L,C]=cs_init(N,nvar,span)
if size(span,1)~=nvar
    span=(ones(nvar,1)*[min(span(:,1)),max(span(:,2))]);
end
L = span(:,2)-span(:,1);
C = span(:,1);
popu = (L*ones(1,N)).*rand(nvar,N)+(C*ones(1,N));
end

%% continuer_gCS, decide if the algorithm keeps going or not
function b=continuer_gCS(hist,i,t,seuil,imax,tmax)
%hist : cost history
%i : scalar, current iteration
%t : scalar, current stopwatch value
%seuil : scalar, threshold
%imax : iteration max.
%tmax : duration max.
b=true;
if hist(size(hist,2))<seuil
    b=false;
end
if i>imax
    b=false;
end
if t>tmax
    b=false;
end
end

%% trier, Sort popu_ini according to popu_imag
function [popu_ini_triee, imag_popu_ini_triee] = trier(popu_ini,...
    imag_popu_ini)
[imag_popu_ini_triee, permut] = sort(imag_popu_ini,'ascend');
popu_ini_triee = popu_ini(:,permut);
end

%% selection
% Extract a part of the population
% Remark: the input population has to be sorted
function [popu_select] = selection(tab,nbre)
popu_select = tab(:,1:nbre);
end

%% clonage
% Clone the population
% Remark: the input population has to be sorted
% (N*floor(be*N)) (total number of clones) = N*ValEntiere(be*N)
function [popu_clone] = clonage(popu_select,be1,N1)
popu_clone = matrix((ones(floor(be1*N1),1)*matrix(popu_select',1,-1))...
    ,N1*floor(be1*N1),-1)';
end

%% gcs_mutation, Mutation
function [popu_mut] = gcs_mutation(popu_clones,nvar1,be1,N1,...
    taux_mut_clones1,span1,C1)
popu_mut = popu_clones.*(1 + (2*rand(nvar1,(N1*floor(be1*N1)))-1).*...
    taux_mut_clones1);
D=span1(:,2)*ones(1,(N1*floor(be1*N1)));
G=C1*ones(1,(N1*floor(be1*N1)));
popu_mut = (popu_mut>=D).*D+(popu_mut<=G).*G+((popu_mut<D)&...
    (popu_mut>G)).*popu_mut;
end

%% inserer
% Insertion of a population in an other
function [popu_nouvelle,imag_popu_nouvelle] = inserer(tab1, tab2,...
    imag_tab2,N1,a1,a2,tfct,argf)
[tri,imag_tri] = trier([tab1,tab2], [costfunc1DV01(tab1,a1,a2,tfct,argf),...
    imag_tab2]);
popu_nouvelle = selection(tri,N1);
imag_popu_nouvelle = selection(imag_tri,N1);
end

%% forcer
% Replace the last "d" cells of tab2 by a random population
% inside tab1, Remark: tab2 has to be sorted
function [popu_nouvelle,imag_popu_nouvelle] = forcer(tab1,tab2,...
    imag_tab2,N1,d1,a1,a2,tfct,argf)
if isempty(tab1)
	popu_nouvelle = tab2;
	imag_popu_nouvelle = imag_tab2;
else
	popu_nouvelle = [tab2(:,1:(N1-d1)),tab1];
	imag_popu_nouvelle = [imag_tab2(:,1:(N1-d1)),...
        costfunc1DV01(tab1,a1,a2,tfct,argf)];
end
end
