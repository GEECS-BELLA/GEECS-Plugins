%%%%%%%%%% PROGRAM ssaPltV02.m %%%%%%%%%%
% SSA plotter ver.2
% Kei Nakamura
% Created on Sunday, November 13th 2005.
% Modified on Tuesday, June 3rd 2008.

% This program reads the log files,
% and plots a SSA signal against a compressor value
% with errorbar (standard deviation).
% Then it generates an output file (name.txt).

clear all
close all

[logNam,logPath] = uigetfile('*.txt','Choose a log file');
[ttl,dat] = fLogReadV02(logNam);   % if modified by excel, use readlog2
ttl

scan = input('scaning parameter number: ');
pltAxs = input('plotting parameter number: ');

% take error out (less than 100 micron), and change to fs
sizDat = size(dat);
skip = 0;
for i=1:sizDat(1)
    if dat(i,pltAxs)>1e-4;      %Threshold
        %subDat(i-skip,:) = [dat(i,scan) 280000*dat(i,pltAxs)];
        subDat(i-skip,:) = [dat(i,scan) dat(i,pltAxs)];
    else skip = skip + 1;
    end
end

minC = min(subDat(:,1))
maxC = max(subDat(:,1))
start = input('input the beginning of the scan: ');
finish = input('input the end of the scan: ');
step = input('step: ');
shift = input('shift[mm]: ');

stpNum = ceil((finish - start)/step+1);
skip = 0;
for i=1:stpNum
    k=find(subDat(:,1)>=start+(i-1)*step&subDat(:,1)<start+i*step); % bin
    ks = size(k);                               % size of bin
    if ks(1,1)~0 ;
        lclDat = subDat(k,:);    % make a subset (bin)
        pltDat(i-skip,:) = statdata(lclDat);    % average and deviation
    else skip = skip + 1;
    end
end
stpNum = stpNum - skip;                        % real step number
pltDat(:,5) = pltDat(:,1) + shift;          % shifted comp
p1 = polyfit(pltDat(:,1),pltDat(:,2),6)     % fit for non-shift
p2 = polyfit(pltDat(:,5),pltDat(:,2),6)     % fit for shifted

pltDat(:,4) = polyval(p1,pltDat(:,1));       % fit result non-shift
pltDat(:,6) = polyval(p2,pltDat(:,5));       % fit result with shift

% to show minimum
finish = finish + shift;
start = start + shift;
stp2 = (finish - start)/1000;
x = [1:1000];
x = x*stp2+start;
y = polyval(p2,x);
[a b] = min(y);
x(b)
y(b)

d = finish - start;
start2 = start - 0.5*d;
finish2 = finish + 0.5*d;
stp3 = (finish2 - start2)/1000;
x2 = [1:1000];
x2 = x2*stp3+start2;
y2 = polyval(p2,x2);

fig1=figure(1);
set(gcf,'color',[1 1 1]);
set(fig1, 'Position', [400 850 400 400]);
set(gca,'fontsize',14);
errorbar(pltDat(:,1),pltDat(:,2),pltDat(:,3),'o')
hold on
plot(pltDat(:,1),pltDat(:,4),'k-')
plot(pltDat(:,5),pltDat(:,6),'r-')
hold off
xlabel(ttl{scan,2})
ylabel('[fs]')
set(gca,'fontsize',14);

f9 = fEscMakerV01('%f',9);            % for output file
s9 = fEscMakerV01('%s',9);
f6 = fEscMakerV01('%f',6);            % for output file
s6 = fEscMakerV01('%s',6);
nameout = strcat('SSA',logNam);
fid=fopen(nameout,'w');
fprintf(fid,s9,'minX','min[fs]','6','5','4','3','2','1','0');
fprintf(fid,f9,x(b),y(b),p2');
fprintf(fid,s6,ttl{scan,2},'width[fs]','deviation','fit','shift-comp','shift-fit');
fprintf(fid,f6,pltDat');
fclose(fid);
