fig2 = figure(2);
set(gcf,'color',[1 1 1]);
set(fig2, 'Position', [1350 450 600 500]);
set(fig2, 'visible', 'on');

aa = double(Scan012_CAM_1BL_ShotControl_002);
%bb=double(Scan004_CAM_LB_AMP1East_003);
%cc=double(Scan008_CAM_LT_AMP1_001);
%dd = double(Scan008_CAM_LB_AMP1West_001);

%%
%subplot(1,2,1)
pcolor(aa)
shading interp
colorbar
axis equal
axis([700 1200 400 800])
%clim([0 45000])
title('tel4')
xlabel('[pixel]')
ylabel('[pixel]')
set(gca,'fontsize',16);

%%
subplot(1,2,2)
pcolor(bb)
shading interp
colorbar
axis equal
axis([1000 1600 600 1350])
title('amp1 east new head')
xlabel('[pixel]')
ylabel('[pixel]')
set(gca,'fontsize',16);

%%
subplot(2,2,3)
pcolor(bb)
shading interp
colorbar
axis equal
axis([200 1000 100 800])
title('amp2 cam')
xlabel('[pixel]')
ylabel('[pixel]')
set(gca,'fontsize',16);

%%
subplot(2,2,4)
pcolor(aa)
shading interp
colorbar
axis equal
%axis([375 725 750 1100])
%axis([350 760 800 1150])
axis([700 1200 400 800])
title('shot control 1 cam')
xlabel('[pixel]')
ylabel('[pixel]')
set(gca,'fontsize',16);
