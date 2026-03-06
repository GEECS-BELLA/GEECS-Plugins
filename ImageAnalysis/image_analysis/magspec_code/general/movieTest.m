clear mov;
mov = avifile('movieTest.avi','quality',70);

fig1 = figure(1);
set(gcf,'color',[1 1 1]);
set(fig,'DoubleBuffer','on');
set(gca,'fontsize',12);
set(fig, 'Position', [30 0 425 300]);

strt = 2854322;
fnsh = 2854355;
for i=strt:fnsh
    fNam = strcat(num2str(i),'tif');
    if exist(fNam)
        a = double(imread(fNam));
        pcolor(a);
        shading interp;
        colorbar;

        frm = getframe(gcf);
        mov = addframe(mov,frm);
    end
end

mov = close(mov);
