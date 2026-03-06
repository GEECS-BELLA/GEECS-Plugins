%% compare charts

%% Revisions
% 2021/12/18 v.1: create

%% initialization
clear all
close all
warning off

fig1 = figure(1);
set(gca,'fontsize',14);
set(gcf,'color',[1 1 1]);
set(fig1, 'Position', [1200 850 1200 650]);

fig2 = figure(2);
set(gca,'fontsize',14);
set(gcf,'color',[1 1 1]);
set(fig2, 'Position', [1200 850 1200 650]);

fig3 = figure(3);
set(gca,'fontsize',14);
set(gcf,'color',[1 1 1]);
set(fig3, 'Position', [1200 850 1200 650]);

fig4 = figure(4);
set(gca,'fontsize',14);
set(gcf,'color',[1 1 1]);
set(fig4, 'Position', [1200 850 1200 650]);

%% input
prjN = 10;   % project number

%%
dir1 = '/Users/KNakamura/Dropbox (Bella Center)/Documents/private/market';
prjList = dir([dir1,filesep,'2*']);

prjName = prjList(prjN).name;   % project name

fList = dir([dir1,filesep,prjName,filesep,'*.csv']);
mntData = zeros(numel(fList),13);   % monthly data for 1 year
mntDataH = zeros(numel(fList),7);   % monthly data for 0.5 year

for ii=1:numel(fList)
    a = readtable([fList(ii).folder,filesep,fList(ii).name]);
    mStp = (numel(a.Close)-1)/12;
    mStp1 = round(1:mStp:numel(a.Close));

    a.Close = a.Close/a.Close(1);   % 1-year normalize
    a.Open = a.Close/a.Close(mStp1(7));   % 0.5 year normalize
    leg = fList(ii).name;
    set(0,'CurrentFigure',fig1)
    plot(a.Date(1:1:end), a.Close(1:end), 'linewidth',2,'DisplayName',leg(1:end-4))
    hold on

    set(0,'CurrentFigure',fig2)
    plot(a.Date(mStp1(7):1:end), a.Open(mStp1(7):1:end), 'linewidth',2,'DisplayName',leg(1:end-4))
    hold on
    %disp([leg(3:end-4) ' ' num2str(a.Close(end)) ' ' num2str(a.Open(end))])

    mntData(ii,:) = a.Close(mStp1);
    mntDataH(ii,:) = a.Open(mStp1(7:end));
end
dateP = a.Date(mStp1);
dateH = a.Date(mStp1(7:end));

set(0,'CurrentFigure',fig1)
hold off
legend('show')
title(prjName)
set(gca,'fontsize',16);

set(0,'CurrentFigure',fig2)
hold off
legend('show')
title(prjName)
set(gca,'fontsize',16);

%% 0.5 year
[~,ind2] = sort(mntDataH(:,end),'descend');   % order
set(0,'CurrentFigure',fig4)
mntDataHS = sum(mntDataH);
mntDataH2 = mntDataH;
for ii=1:7
    mntDataH(:,ii) = 100*mntDataH(:,ii)/mntDataHS(ii);
end
for ii=1:numel(fList)
    jj = ind2(ii);
    leg = fList(jj).name;
    plot(dateH,mntDataH(jj,:),'linewidth',2,'DisplayName',leg(1:end-4))
    hold on
    mntDataH2(ii,:) = mntDataH(jj,:); % sorted
    disp(leg(1:end-4))
end
hold off
legend('show')
title([prjName, ': 0.5-year'])
set(gca,'fontsize',16);
disp(' ')
writematrix(mntDataH2,[fList(ii).folder,filesep,'textH.txt'])

%% 1 year
[~,ind] = sort(mntData(:,end),'descend');   % order
set(0,'CurrentFigure',fig3)
mntDataS = sum(mntData);
mntData2 = mntData;
for ii=1:13
    mntData(:,ii) = 100*mntData(:,ii)/mntDataS(ii);
end
for ii=1:numel(fList)
    %jj = ind(ii); % separate sort
    jj = ind2(ii); % same sort
    leg = fList(jj).name;
    plot(dateP,mntData(jj,:),'linewidth',2,'DisplayName',leg(1:end-4))
    hold on
    mntData2(ii,:) = mntData(jj,:); % sorted
    disp(leg(1:end-4))
end
hold off
legend('show')
title([prjName, ': 1-year'])
set(gca,'fontsize',16);

writematrix(mntData2,[fList(ii).folder,filesep,'text.txt'])
