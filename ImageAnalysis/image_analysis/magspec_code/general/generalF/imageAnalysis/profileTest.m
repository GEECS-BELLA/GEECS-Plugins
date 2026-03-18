function profileTest(yScl,yPrf)
%PROFILETEST    Create plot of datasets and fits
%   PROFILETEST(YSCL,YPRF)
%   Creates a plot, similar to the plot in the main curve fitting
%   window, using the data that you provide as input.  You can
%   apply this function to the same data you used with cftool
%   or with different data.  You may want to edit the function to
%   customize the code and this help message.
%
%   Number of datasets:  1
%   Number of fits:  1


% Data from dataset "yPrf vs. yScl":
%    X = yScl:
%    Y = yPrf:
%    Unweighted
%
% This function was automatically generated on 31-May-2007 13:49:14

% Set up figure to receive datasets and fits
f_ = clf;
figure(f_);
set(f_,'Units','Pixels','Position',[611 267 672 469]);
legh_ = []; legt_ = {};   % handles and text for legend
xlim_ = [Inf -Inf];       % limits of x axis
ax_ = axes;
set(ax_,'Units','normalized','OuterPosition',[0 0 1 1]);
set(ax_,'Box','on');
axes(ax_); hold on;


% --- Plot data originally in dataset "yPrf vs. yScl"
yScl = yScl(:);
yPrf = yPrf(:);
h_ = line(yScl,yPrf,'Parent',ax_,'Color',[0.333333 0 0.666667],...
     'LineStyle','none', 'LineWidth',1,...
     'Marker','.', 'MarkerSize',12);
xlim_(1) = min(xlim_(1),min(yScl));
xlim_(2) = max(xlim_(2),max(yScl));
legh_(end+1) = h_;
legt_{end+1} = 'yPrf vs. yScl';

% Nudge axis limits beyond data limits
if all(isfinite(xlim_))
   xlim_ = xlim_ + [-1 1] * 0.01 * diff(xlim_);
   set(ax_,'XLim',xlim_)
end


% --- Create fit "fit 1"
fo_ = fitoptions('method','NonlinearLeastSquares','Robust','On','Lower',[0  -Inf 0 0 ],'MaxFunEvals',395,'MaxIter',380);
ok_ = ~(isnan(yScl) | isnan(yPrf));
st_ = [1.35 10 0.4264515162825 0.15 ];
set(fo_,'Startpoint',st_);
ft_ = fittype('a1*exp(-((x-b1)/c1)^2)+d1' ,...
     'dependent',{'y'},'independent',{'x'},...
     'coefficients',{'a1', 'b1', 'c1', 'd1'});

% Fit this model using new data
cf_ = fit(yScl(ok_),yPrf(ok_),ft_ ,fo_);

% Or use coefficients from the original fit:
if 0
   cv_ = {1.165116813593, 10.24615088826, 4.772018095255, 0.2116687656985};
   cf_ = cfit(ft_,cv_{:});
end

% Plot this fit
h_ = plot(cf_,'fit',0.95);
legend off;  % turn off legend from plot method call
set(h_(1),'Color',[1 0 0],...
     'LineStyle','-', 'LineWidth',2,...
     'Marker','none', 'MarkerSize',6);
legh_(end+1) = h_(1);
legt_{end+1} = 'fit 1';

% Done plotting data and fits.  Now finish up loose ends.
hold off;
h_ = legend(ax_,legh_,legt_,'Location','NorthEast');
set(h_,'Interpreter','none');
xlabel(ax_,'');               % remove x label
ylabel(ax_,'');               % remove y label
