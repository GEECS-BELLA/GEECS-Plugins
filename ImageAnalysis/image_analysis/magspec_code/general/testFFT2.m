cenWvl = 800;   % central wavelength [nm]
sigWvl = 15;    % sigma of spectrum [nm]
minWvl = 300;   % minimum wavelength [nm]
maxWvl = 1100;  % max. wavelength [nm]
pixN = 256;     % number of pixel

sigLsr = 50;    % sigma of laser pulse [fs]


wvl = [minWvl:(maxWvl-minWvl)/(pixN-1):maxWvl];   % wavelength [nm]
spcW = exp(-(wvl-cenWvl).^2/(2*sigWvl)^2);  % power spec of wavelength

frq = 299.8./wvl;    % frequency [PHz]
minFrq = min(frq); maxFrq = max(frq);   % min and max freq [PHz]
Dv = maxFrq - minFrq;           % Delta v
dv = Dv/(pixN-1);               % delta v
frqF = [minFrq:dv:maxFrq]; % frequency for FFT [PHz]
spcF = interp1(frq,spcW,frqF);  % spectrum for FFT []

spcT = fftshift(fft(spcF));
pSpcT = abs(spcT).^2;
dt = 1/(maxFrq - minFrq);       % dt [fs]
Dt = dt*(pixN-1);               % Dt [fs]
tim = [0:dt:Dt];

disp([dv Dv dt Dt])

figure(1)
subplot(2,3,1)
plot(wvl,spcW)
subplot(2,3,2)
plot(frq,spcW)
subplot(2,3,3)
plot(spcF)
subplot(2,3,4)
plot(real(spcT),'o')
subplot(2,3,5)
plot(tim,pSpcT)
