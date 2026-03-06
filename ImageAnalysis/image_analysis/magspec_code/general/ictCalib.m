scanS = '004';
fld = ['/Users/kee/Dropbox (Bella Center)/Documents/data/BELLA/20_0921 ICT calib/scans/Scan',scanS,'/PLIF-Tek DPO scope'];
fName = [fld,filesep,'Scan',scanS,'_PLIF-Tek DPO scope_001.dat'];
h5disp(fName,'/wfm_group0/axes/axis0');
ictD1 = fReadNIWaveHdf5(fName,1);
ictD2 = fReadNIWaveHdf5(fName,3);
ictD3 = fReadNIWaveHdf5(fName,5);

ictAx = h5info(fNameI,'/wfm_group0/axes/axis0');    % ict x-axis
ictAxStrt = ictAx.Attributes(2).Value;  % x-axis start value
ictAxIncr = ictAx.Attributes(3).Value;  % x-axis increment
ictAxLast = ictAxStrt+ictAxIncr*(numel(ictD1)-1); % x-axis last
ictXns = 1e9*[ictAxStrt:ictAxIncr:ictAxLast]; % ict x axis [ns]


plot(ictXns,ictD1)
hold on
plot(ictXns,ictD2)
plot(ictXns,ictD3)
hold off
axis([-50 100 0 1])
