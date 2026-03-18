function hst = fGetHst2DV01(x,y,img)
% Create histogram like data for fitting functions
%
% hst = fGetHstV01(x,y)
% x: axis information (x axis)
% y: counts for each point (yaxis)
% img: source image
% hst: formatted for fitting function input
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get histogram for 2d Gaussian fit, ver.1
%
% Kei Nakamura
% Created on Friday, February 12th 2010.
% Modified on
%
% ver.1: created
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% main body
aaa = round(img);
[szy,szx] = size(aaa);
hst = zeros(sum(sum(aaa)),2);
zz = 1;
for xx=1:szx;   % x
    for yy=1:szy;   % y
        while aaa(yy,xx)>0
            hst(zz,1) = x(xx);
            hst(zz,2) = y(yy);
            zz = zz + 1;
            aaa(yy,xx) = aaa(yy,xx) -1;
        end
    end
end
