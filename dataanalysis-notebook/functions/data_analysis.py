"""
Useful functions for analyzing dataframe in general
"""

from scipy.interpolate import UnivariateSpline
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def threshold(data, feature, min=None, max=None):
    '''Remove datasets with output value outside of min < out < max
    feature: string name of the feature, or the data itself
    '''
    if isinstance(feature, str) and feature in list(data):
        y = data[feature]
    else:
        print('2nd arg is data')
        y = feature

    if min != None:
        data = data[data[y]>=min]
    if max != None:
        data = data[y<=max]

    return data_2

def df_subtractfit(df, col, deg = 0, calib = 1):
    '''Subtract the fitted data. if deg=0, mean is set to zero
    col: list of column name
    calib: calibration
    '''
    t = list(range(len(df))) # get a list of consecutive numbers
    
    for i in range(len(col)):
        df[col[i]] = (df[col[i]] - fit_data(t,df[col[i]],deg))*calib
    return df

def find_fwhm_x(x, y):
    '''Find x values where the signals are Half maximum
    return:x1, x2, where x2-x2 is the FWHM
    To plot:
        import pylab as pl
        pl.plot(x, y)
        pl.axvspan(r1, r2, facecolor='g', alpha=0.5)
    '''
    # create a spline of x and blue-np.max(blue)/2 
    spline = UnivariateSpline(x, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots
    return r1, r2

def correlation_adj(df):
    '''
    Calculate a correlation coefficient between the adjacent rows.
    df: Dataframe
    return: list of correlation coefficients
    '''
    corr_diff=[1]
    for i in range(len(df.columns)-1):
        corr_diff.append(df.corr().values[i][i+1])
    return corr_diff

def residual_rms_list(df, i_start=0, calib=None, fittype='linear regression'):
    '''
    Get a list of residual standard deviation between adjacent columns of dataset.
    For a two columns next to each other, do a linear fit, poject the data into the
    first data set (out of two), measure the RMS of the projected data on the axis
    of the second dataset.
    calib: list of calibration for each column of data
    type: 'linear regression' or 'PCA'
    '''
    res_list = [0]
    for i in range(i_start, len(df.columns)-1):
        x = df[df.columns[i_start]]
        y = df[df.columns[i+1]]
        if fittype=='linear regression':
            y_res = y - fit_data(x,y,1)
        if fittype=='PCA':
            y_res = y - fit_PCA(x, y)
        if calib:
            res_list.append(y_res.std()*calib[i])
        else:
            res_list.append(y_res.std())

    return res_list

def fit_PCA(x,y):
    '''Fit with PCA.
    return: x, fit value corresponding to x
    '''
    xy = np.array([x,y]).T
    pca = PCA(n_components=1)
    xy_pca = pca.fit_transform(xy)
    xy_n = pca.inverse_transform(xy_pca)
    
    #fit the PCA fitline
    z = np.polyfit(xy_n[:,0],xy_n[:,1],1)
    return x*z[0]+z[1]

def residual_rms(x,y):
    #linear fit
    z = np.polyfit(x, y, 1)
    print('slope:',z[0])
    #Projection of points to x axis
    y_res = y - (x*z[0]+z[1])
    return y_res.std()
    
def fit_data(t,signal, deg):
    '''fit the signal, return the fitted data
    (t,signal): dataset
    deg: degree of the fitting polynomial 
    '''
    z = np.polyfit(t, signal,deg)
    p = np.poly1d(z)
    return p(t) 


def df_zscore(df, cols):
    '''Calculate zscore of the dataframe for all cells.
    This function works for dataframe including NaNs.
    (scipy.stats.zscore does not work for NaNs. It returns NaNs for the whole col.)
    df: data frame
    return: data frame with zscores. NaN cells return NaN.
    '''
    df_zscore=pd.DataFrame()
    for col in cols:
        df_zscore[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)   
    return df_zscore

def df_outlier2none(df,std=1,columns = None):
    '''
    Change the volume outside of a standard deviation in a dataframe to None
    df: dataframe
    columns: list of the colum names to consider the change
    std: how many sigma to include in the data?
    '''
    if columns == None:
        columns = df.columns
    df_z = df_zscore(df, columns)
    df_isnan = ~(np.abs(df_z) < float(std)) #if data is outside of std_dev*sigma
    
    for col in columns:
        df.loc[df_isnan[col], col]=None
    return df

def idx_df_nan(df, col):
    '''
    Find a index where values contain Nan
    '''
    return df[col].index[df[col].apply(np.isnan)]


def rm_zero(data, yname):
    '''Remove dataset with output value 0
    '''
    data = data[data[yname] != 0]
    return data