function imgOut = fCmrSigV01(img,sgm)
%% extend dynamic range for camera sigma
%
% imgOut = fCmrNoiseV01(img,cmrThre,cmrSig,sttNmb)
% imgOut: output image
%
% img:  input image
% cmrSig: camera noise in sigma
%

%% Written by Kei Nakamura
%
% 2015/10/9 ver.1: Created

%% used in
% bellaFrogFitMSV05

%% main body
imgLgc = img<(3*sgm);   % logic for correctoin
imgPrt = img.*imgLgc;   % part of image

x = linspace(-3*sgm,3*sgm,1000); % explore +/- 3 sigma, also means actual signal
fx = (x(2)-x(1))*exp(-x.^2/(2*sgm^2))/(sgm*sqrt(2*pi)); % distribution function
avrg = x*0; % allocation, average count
%cdf = 0.5*(1+erf((x-mu)/(sgm*sqrt(2))));

for jj=1:1000
    prb = fx(1001-jj:1000);    % probability of being >0
    count = x(1001-jj:1000)+x(jj);  % count for it.
    avrg(jj) = sum(prb.*count); % average count
end

imgCrr = interp1(avrg,x,imgPrt);   % correction
imgOut = img.*~imgLgc + imgCrr.*imgLgc;
imgOut = imgOut + 1*sgm;    % bias
imgOut = imgOut.*(imgOut>0);    % negative ->0


% figure(10)
% subplot(2,2,1)
% pcolor(img)
% shading interp
% colorbar
%
% subplot(2,2,2)
% pcolor(imgOut)
% shading interp
% colorbar
%
% subplot(2,2,3)
% plot(img(120,:),'b-')
% hold on
% plot(imgOut(120,:),'r-')
% hold off
% axis([0 200 -10 550])
%
% subplot(2,2,4)
% plot(avrg,x)
% hold on
% plot(x,x)
% hold off
