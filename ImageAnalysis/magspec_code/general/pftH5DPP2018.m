% This program to make plots for DPP2018,
% with simulation results from Maxence

fig1 = figure(1);
set(gca,'fontsize',14);
set(gcf,'color',[1 1 1]);
set(fig1, 'Position', [1200 850 700 450]);

dir1 = '/Users/kee/Dropbox (Bella Center)/Documents/myPresentations/1811DPP/pft_for_kei/';
fNam1 = [dir1,'without_plasma_nogvd.h5'];
fNam2 = [dir1,'with_plasma_a0_0.1.h5'];
fNam3 = [dir1,'with_plasma_a0_1.0.h5'];
fNam4 = [dir1,'with_plasma_a0_3.0.h5'];
fNam5 = [dir1,'with_plasma_a0_3.0_gvd0.h5'];
fNam6 = [dir1,'with_plasma_a0_3.0_gvd14.h5'];
fNam7 = [dir1,'with_plasma_a0_3.h5'];

%%
pf1 = fReadPftH5(fNam1);
pf2 = fReadPftH5(fNam5);    % gvd 0
pf3 = fReadPftH5(fNam4);    % gvd 7
pf4 = fReadPftH5(fNam6);    % gvd 14

%% fig 1-1
figure(1)
plot(0)
fOut = [dir1,'fig1'];
subplot(2,1,2)
plot(pf1.lsrZ,-pf1.pft,'k-','linewidth',2)
grid on
axis([0 0.02 -0.22 0.22])
%legend('vacuum','gvd=0','gvd=70fs^2','gvd=140fs^2','location','southwest');
xlabel('z [m]')
ylabel('pft angle [rad]')
set(gcf,'color',[1 1 1]);
set(gca,'fontsize',16);

subplot(2,1,1)
plot(pf1.lsrZ,1e6*pf1.wdt,'k-','linewidth',2)
grid on
%legend('vacuum','gvd=0','gvd=70fs^2','gvd=140fs^2','location','southwest');
title('vacuum propagation')
xlabel('z [m]')
ylabel('laser width [um]')
axis([0 0.02 40 100])
set(gca,'fontsize',16);
drawnow
saveSameSize(fig1,'format','png','renderer','opengl','file',fOut);

%% fig1-2
figure(1)
plot(0)
fOut = [dir1,'fig2'];
subplot(2,1,2)
plot(pf1.lsrZ,-pf1.pft,'k--','linewidth',1)
hold on
plot(pf2.lsrZ,-pf2.pft,'-','linewidth',2)
grid on
hold off
axis([0 0.02 -0.22 0.22])
legend('vacuum','plasma','location','southwest');
xlabel('z [m]')
ylabel('pft angle [rad]')
set(gcf,'color',[1 1 1]);
set(gca,'fontsize',16);

subplot(2,1,1)
plot(pf1.lsrZ,1e6*pf1.wdt,'k--','linewidth',1)
hold on
plot(pf2.lsrZ,1e6*pf2.wdt,'-','linewidth',2)
grid on
hold off
legend('vacuum','plasma','location','southwest');
xlabel('z [m]')
ylabel('laser width [um]')
title('2cm-long uniform plasma')
axis([0 0.02 40 100])
set(gca,'fontsize',16);
drawnow
saveSameSize(fig1,'format','png','renderer','opengl','file',fOut);

%%
figure(1)
plot(0)
fOut = [dir1,'fig3'];
subplot(2,1,2)
plot(pf1.lsrZ,-pf1.pft,'k--','linewidth',1)
hold on
plot(pf2.lsrZ,-pf2.pft,'-','linewidth',2)
%plot(pf3.lsrZ,pf3.pft,'o')
plot(pf4.lsrZ,-pf4.pft,'x','linewidth',2)
grid on
hold off
axis([0 0.02 -0.22 0.22])
xlabel('z [m]')
ylabel('pft angle [rad]')
set(gcf,'color',[1 1 1]);
set(gca,'fontsize',16);

subplot(2,1,1)
plot(pf1.lsrZ,1e6*pf1.wdt,'k--','linewidth',1)
hold on
plot(pf2.lsrZ,1e6*pf2.wdt,'-','linewidth',2)
%plot(pf3.lsrZ,pf3.wdt,'-','linewidth',2)
plot(pf4.lsrZ,1e6*pf4.wdt,'x','linewidth',2)
grid on
hold off
xlabel('z [m]')
ylabel('laser width [um]')
title('2cm-long uniform plasma')
%legend('vacuum','gvd=0','gvd=70fs^2','gvd=140fs^2','location','north');
legend('vacuum','gvd=0','gvd=140fs^2','location','north');
axis([0 0.02 40 100])
set(gca,'fontsize',16);
drawnow
saveSameSize(fig1,'format','png','renderer','opengl','file',fOut);
