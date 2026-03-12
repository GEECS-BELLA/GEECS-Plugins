%% fast profiler analysls

%% Written by Kei Nakamura
% 2020/3/26 ver.1:, created

%% initialization
clear all

%% input
dr1 = '/Volumes/BDNA1/data/PWlaserData/Y2021/03-Mar/21_0326/FastProfiler';
fileNmb = 2;
smplF = 1000;   % sampling f [Hz]
roi2 = [2400,2700];

fList = dir([dr1,filesep,'*.txt']); % file list
bgTbl = [fList(2).folder,filesep,fList(2).name];  % bg1, table
tbl = [fList(3).folder,filesep,fList(3).name];

[t,f,cx1,cy1,fx1,fy1] = fFastProfilerRead(bgTbl,smplF);
[~,~,cx2,cy2,fx2,fy2] = fFastProfilerRead(tbl,smplF);
[~,~,cx1b,cy1b,fx1b,fy1b] = fFastProfilerRead(bgTbl,smplF,roi2);
[tb,fb,cx2b,cy2b,fx2b,fy2b] = fFastProfilerRead(tbl,smplF,roi2);

figure(1)

subplot(3,2,1)
plot(t,cx1,'.')

subplot(3,2,2)
plot(f,fx1)
aa = axis;
axis([aa(1:3),max(fx1(20:end))])

subplot(3,2,3)
plot(t,cx2,'.')
hold on
plot(t(roi2(1):roi2(2)),cx2(roi2(1):roi2(2)),'.')
hold off

subplot(3,2,4)
plot(f,fx2)
aa = axis;
axis([aa(1:3),max(fx2(20:end))])

subplot(3,2,5)
plot(fb,fx2b)
axis([aa(1:3),max(fx2b(20:end))])
hold on
plot(fb,fx1b)
hold off

subplot(3,2,6)
plot(fb,fx2b - fx1b)
axis([aa(1:3),max(fx2b(20:end))])

figure(2)

subplot(3,2,1)
plot(t,cy1,'.')

subplot(3,2,2)
plot(f,fy1)
aa = axis;
axis([aa(1:3),max(fy1(20:end))])

subplot(3,2,3)
plot(t,cy2,'.')
hold on
plot(t(roi2(1):roi2(2)),cy2(roi2(1):roi2(2)),'.')
hold off

subplot(3,2,4)
plot(f,fy2)
aa = axis;
axis([aa(1:3),max(fy2(20:end))])

subplot(3,2,5)
plot(fb,fy2b)
axis([aa(1:3),max(fy2b(20:end))])
hold on
plot(fb,fy1b)
hold off

subplot(3,2,6)
plot(fb,fy2b - fy1b)
axis([aa(1:3),max(fy2b(20:end))])
