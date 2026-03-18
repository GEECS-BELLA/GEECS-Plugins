%% PROGRAM datPltV01.m %%%%%%%%%%
% This program reads a log file, and plots data with errorbar (standard deviation).
%
% data plotter ver.1
% Kei Nakamura
% Created on Wednesday, November 25th 2009.
% Modified on

%% main body
clear all
close all

[logNam,logPath] = uigetfile('*.txt','Choose a log file');
[ttl,dat] = fLogReadV02([logPath,logNam]);   % if modified by excel, use readlog2
disp(ttl);  % display titles

scan = input('scaning parameter number(x): ');
pltAxs = input('plotting parameter number(y): ');

% % take error out (less than 100 micron), and change to fs
% sizDat = size(dat);
% skip = 0;
% for i=1:sizDat(1)
%     if dat(i,pltAxs)>1e-4;      %Threshold
%         subDat(i-skip,:) = [dat(i,scan) 280000*dat(i,pltAxs)];
%     else skip = skip + 1;
%     end
% end


minC = min(dat(:,scan));
maxC = max(dat(:,scan));
disp(['minimum x: ',num2str(minC),', max x: ',num2str(maxC)]);

start = input('input minimum x to plot: ');
finish = input('input max x to plot: ');
step = input('input step for x: ');

stpNum = ceil((finish - start)/step+1);
skip = 0;
for i=1:stpNum
    k=find(dat(:,scan)>=start+(i-1)*step&dat(:,scan)<start+i*step); % bin
    ks = size(k);                               % size of bin
    if ks(1,1)~0 ;
        lclDat = dat(k,scan);    % make a subset (bin)
        lclDat(:,2) = dat(k,pltAxs);    % make a subset (bin)
        pltDat(i-skip,:) = fStatDataV01(lclDat);    % average and deviation
    else skip = skip + 1;
    end
end
stpNum = stpNum - skip;                        % real step number

% % to show minimum
% finish = finish + shift;
% start = start + shift;
% stp2 = (finish - start)/1000;
% x = [1:1000];
% x = x*stp2+start;
% y = polyval(p2,x);
% [a b] = min(y);
% x(b)
% y(b)
%
% d = finish - start;
% start2 = start - 0.5*d;
% finish2 = finish + 0.5*d;
% stp3 = (finish2 - start2)/1000;
% x2 = [1:1000];
% x2 = x2*stp3+start2;
% y2 = polyval(p2,x2);

fig1=figure(1);
set(gcf,'color',[1 1 1]);
set(fig1, 'Position', [400 850 400 400]);
set(gca,'fontsize',14);
errorbar(pltDat(:,1),pltDat(:,2),pltDat(:,6),pltDat(:,8),'o')
