#
# Functions for parameter grid search, model diagnostics and
# forecasting.
#

import os
import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, probplot
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse, meanabs
from statsmodels.tsa.stattools import acf


#
# Grid Search Functions
#

def gs_sarima(series, param, s_param):
    '''Perform grid search to return optimal parameters.

    Parameters
    ----------
    series : pd.Series
        Series of counts with DateTime index.
    param : tuple
        (p, d, q) tuple of ARIMA parameters.
    s_param : tuple
        (P, D, Q)S tuple of seasonal ARIMA parameters.

    Results
    -------
    df_results : pd.DataFrame
        Data Frame of parameters and AIC score.
    '''
    results = []

    for p in param:
        for s_p in s_param:
            sarima = sm.tsa.statespace.SARIMAX(
                series,
                order=p,
                seasonal_order=s_p,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            mod_results = sarima.fit()
            if not mod_results.mle_retvals['converged']:
                print(
                    'Warning: Maximum Likelihood optimization failed to '
                    'converge on {} {}.'.format(p, s_p)
                )
            results.append([p, s_p, mod_results.aic])
    return (
        pd.DataFrame(
            results,
            columns=['pdq', 'seasonal_pdq', 'aic']
        ).sort_values('aic', ascending=True)
            .reset_index(drop=True)
    )


#
# Figure Functions
#

def save_fig(fig_name, dir_figures):
    '''Save matplotlib figures to figure directory.

    Parameters
    ----------
    fig_name : str
        Name of saved figure.
    dir_figures : str
        Relative location of figures directory.

    Returns
    -------
    None
    '''
    fig_format = 'png'
    fig_file = os.path.join(dir_figures, '{}.{}'.format(fig_name, fig_format))
    plt.savefig(fig_file, format=fig_format, bbox_inches='tight')


#
# Model Diagnostic Functions
#

def plot_sarima_resid(ax, resid, alpha_mpl):
    '''Plot SARIMA residuals.

    Parameters
    ----------
    ax : Axes class
        Matplotlib axes subplot.
    resid : pd.Series
        Fitted model residuals.
    alpha_mpl : float
        Alpha parameter for matplotlib plot.

    Returns
    -------
    ax : Axes class
        Matplotlib axes subplot.
    '''
    ax.plot(resid, color='C0', alpha=alpha_mpl)
    ax.axhline(linewidth=1, color='C3', alpha=0.5, label='Zero')
    ax.axhline(
        resid.mean(),
        linewidth=1,
        linestyle='--',
        color='#ae017e',
        alpha=0.5,
        label='Mean',
    )
    ax.set_xlabel('Years')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals')
    ax.legend()

    return ax


def plot_sarima_hist(ax, resid, alpha_mpl):
    '''Plot histogram of SARIMA residuals.

    Parameters
    ----------
    ax : Axes class
        Matplotlib axes subplot.
    resid : pd.Series
        Fitted model residuals.
    alpha_mpl : float
        Alpha parameter for matplotlib plot.

    Returns
    -------
    ax : Axes class
        Matplotlib axes subplot.
    '''
    ax.hist(resid, density=True, color='C0', alpha=alpha_mpl)

    x_min, x_max = ax.get_xlim()
    x_density = np.linspace(x_min, x_max, 100)

    kde = gaussian_kde(resid)
    ax.plot(x_density, kde(x_density), color='C1', alpha=0.65, label='KDE')

    mu, std = norm.fit(resid)
    pdf = norm.pdf(x_density, mu, std)
    ax.plot(x_density, pdf, color='C2', alpha=0.65,
            label='Normal Distribution')

    ax.set_xlabel('Residuals')
    ax.set_title('Residuals Distribution')
    ax.legend()

    return ax


def plot_sarima_prob_plot(ax, resid, alpha_mpl):
    '''Plot probability plot of SARIMA residuals.

    Parameters
    ----------
    ax : Axes class
        Matplotlib axes subplot.
    resid : pd.Series
        Fitted model residuals.
    alpha_mpl : float
        Alpha parameter for matplotlib plot.

    Returns
    -------
    ax : Axes class
        Matplotlib axes subplot.
    '''
    (osm, osr), (slope, intercept, __) = probplot(resid, dist='norm', fit=True)

    ax.plot(
        osm,
        osr,
        marker='o',
        markerfacecolor='C0',
        markeredgecolor='C0',
        alpha=alpha_mpl,
        linestyle='',
    )
    ax.plot(osm, (slope * osm + intercept), color='C3', alpha=0.65)

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Values')
    ax.set_title('Residuals Probability Plot')

    return ax


def plot_sarima_correlogram(ax, resid):
    '''Plot correlogram of SARIMA residuals.

    Parameters
    ----------
    ax : Axes class
        Matplotlib axes subplot.
    resid : pd.Series
        Fitted model residuals.

    Returns
    -------
    ax : Axes class
        Matplotlib axes subplot.
    '''
    n_lag = 24
    x_acf, ci = acf(resid, nlags=n_lag, fft=False, alpha=0.05)
    lags = range(n_lag + 1)

    ax.vlines(lags, [0], x_acf[:n_lag + 1], color='C3')
    ax.axhline(color='C3')
    ax.scatter(lags, x_acf[:n_lag + 1], color='C0')
    ax.fill_between(
        lags[1:],
        ci[1:, 0] - x_acf[1:],
        ci[1:, 1] - x_acf[1:],
        color='C0',
        alpha=0.15,
        label='Confidence Interval',
    )

    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    ax.set_title('Residuals Autocorrelation')
    ax.legend()
    return ax


def sarima_diagnostics(resid, alpha_mpl):
    '''Plot residual diagnostics.

    Customize plots for visual consistency across plots in notebook.

    Parameters
    ----------
    resid : pd.Series
        Fitted model residuals.
    alpha_mpl : float
        Alpha parameter for matplotlib plot.

    Returns
    -------
    fig : matplotlib figure object
    '''
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 12),
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    ax0 = plot_sarima_resid(ax0, resid, alpha_mpl)
    ax1 = plot_sarima_hist(ax1, resid, alpha_mpl)
    ax2 = plot_sarima_prob_plot(ax2, resid, alpha_mpl)
    ax3 = plot_sarima_correlogram(ax3, resid)
    return fig


#
# Forecasting Functions
#

def forecast_in_sample(model):
    '''Return in-sample forecast and diagnostic values.

    Parameters
    ----------
    model : statsmodels SARIMAX object
        Object containing fitted SARIMAX results.

    Returns
    -------
    fcast_in : pd.Series of floats
        In-sample forecast.
    ci_in : pd.DataFrame of floats
        In-sample confidence intervals.
    resid_in : pd.Series of floats
        In-sample residuals.
    '''
    pred_in = model.get_prediction(dynamic=False)
    return pred_in.predicted_mean, pred_in.conf_int(), model.resid


def forecast_out_sample(model, dynamic):
    '''Return out-of-sample forecast and diagnostic values.

    Parameters
    ----------
    model : statsmodels SARIMAX object
        Object containing fitted SARIMAX results.
    dynamic : boolean
        Indicates whether forecast should be one-step-ahead or dynamic.

    Returns
    -------
    fcast_out : pd.Series of floats
        Out-of-sample forecasts.
    se_out : pd.Series of floats
        Out-of-sample standard errors.
    ci_out : pd.DataFrame of floats
        Out-of-sample confidence intervals.
    '''
    if dynamic:
        n_fore = 24
    else:
        n_fore = 12

    pred_out = model.get_forecast(n_fore, dynamic=dynamic)
    return pred_out.predicted_mean, pred_out.se_mean, pred_out.conf_int()


def plot_in_sample_forecast(fcast, ci, resid, df):
    '''Plot results of in-sample forecast.

    Parameters
    ----------
    fcast : pd.Series of floats
        In-sample forecast values.
    ci : pd.DataFrame of floats
        In-sample confidence intervals.
    resid : pd.Series of floats
        In-sample residuals.
    df : pd.DataFrame
        Data Frame with datetime index and aggregated requests by month.

    Returns
    -------
    fig : matplotlib figure
        Two plots of in-sample forecast and residuals.
    '''
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.3)

    # Predictions
    ax0.scatter(df.index, df['count'], color='darkblue', label='Observed')
    ax0.plot(fcast, color='C0', label='In-Sample Prediction')
    ax0.fill_between(
        ci.index,
        ci.iloc[:, 0],
        ci.iloc[:, 1],
        color='C0',
        alpha=0.15,
        label='Confidence Interval',
    )

    ax0.set_xlabel('Years')
    ax0.set_ylabel('Requests')
    ax0.set_title('In-Sample One-Step-Ahead Prediction')
    ax0.legend(loc='lower right');

    # Residuals
    ax1.plot(resid, color='C0', label='Error')
    ax1.axhline(linewidth=1, color='C3', alpha=0.5, label=None)
    ax1.axhline(
        resid.mean(),
        linewidth=1,
        linestyle='--',
        color='C1',
        alpha=0.5,
        label='Mean',
    )

    ax1.set_xlabel('Years')
    ax1.set_ylabel('Residuals')
    ax1.set_title('In-Sample Residuals')
    ax1.legend(loc='upper right')
    return fig


def plot_out_sample_forecast(df_train, df_test, dynamic, fcast):
    '''Plot results of out-of-sample forecast.

    Parameters
    ----------
    df_train : pd.DataFrame
        Data Frame of training set data with datetime index and
        aggregated requests by month.
    df_test : pd.DataFrame
        Data Frame of test set data with datetime index and
        aggregated requests by month.
    dynamic : boolean
        Indicates whether forecast should be one-step-ahead or dynamic.
    fcast : tuple(pd.Series of floats, pd.Series of floats,
    pd.DataFrame of floats, pd.DataFrame of floats)
        In-sample forecasts; out-of-sample forecasts; in-sample
        confidence intervals; and out-of-sample confidence intervals.

    Returns
    -------
    fig : matplotlib figure
        Plot of in-sample forecast.
    '''
    fcast_in = fcast[0]
    fcast_out = fcast[1]
    ci_in = fcast[2]
    ci_out = fcast[3]

    if dynamic:
        fore_type = 'Dynamic'
    else:
        fore_type = 'One-Step-Ahead'

    mask_yr = '2014'

    fig, ax = plt.subplots(figsize=(12, 5))

    if not dynamic:
        ax.scatter(
            df_train.loc[mask_yr:].index,
            df_train.loc[mask_yr:]['count'],
            color='darkblue',
            label='Observed'
        )
        ax.scatter(
            df_test.index,
            df_test['count'],
            color='darkblue',
            label=None,
        )
    ax.plot(fcast_in, color='C0', label='In-Sample Forecast')
    ax.plot(fcast_out, color='C1', label='Out-of-Sample Forecast')
    ax.fill_between(
        ci_in.index,
        ci_in.iloc[:, 0],
        ci_in.iloc[:, 1],
        color='C0',
        alpha=0.15,
        label='Confidence Interval',
    )
    ax.fill_between(
        ci_out.index,
        ci_out.iloc[:, 0],
        ci_out.iloc[:, 1],
        color='C1',
        alpha=0.15,
        label='Confidence Interval',
    )

    ax.set_xlim([mask_yr, None])
    ax.set_ylim([-5000, None])
    ax.set_xlabel('Years')
    ax.set_ylabel('Requests')
    ax.set_title('Out-of-Sample {} Forecast'.format(fore_type))
    ax.legend(loc='lower right')
    return fig


def eval_metrics(forecast, observed):
    '''Return forecast evaluation metrics.

    Parameters
    ----------
    forecast : pd.Series
        Forecasted values.
    observed : pd.Series
        Observed values.

    Return
    ------
    mae : float
        Mean Absolute Error metric.
    rmserr : float
        Root Mean Squared Error metric. Named rmserr to avoid
        conflicting with statsmodels rmse function.
    '''
    return meanabs(forecast, observed), rmse(forecast, observed), (
        ((forecast - observed).abs() / observed).mean()) * 100


def plot_eval_metrics(df, mae, rmserr, request_type, sample_type):
    '''Plot model evaluation metrics against min and max of dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame fitted by model.
    mae : float
        Mean Absolute Error of fitted model.
    rmserr : float
        Root Mean Squared Error of fitted model. Named rmserr to avoid
        conflicting with statsmodels rmse function.
    request_type : str
        Either 'Opened' or 'Closed' to indicate the dataset used for the
        fitted model.
    sample_type : str
        Sample type of forecast; either 'In' or 'Out'.

    Returns
    -------
    fig : matplotlib figure
        Plot of MAE and RMSE against min and max of dataset.
    '''
    fig, ax = plt.subplots(figsize=(12, 2.5))

    x_min = df['count'].min()
    x_max = df['count'].max()
    y = [0]
    x_mae = mae
    x_rmse = rmserr

    ax.hlines(y, x_min, x_max, linewidth=1, color='C3', alpha=0.5,
              label=None)
    ax.scatter([x_mae], y, color='C1', s=75, label='MAE')
    ax.scatter([x_rmse], y, color='C0', s=75, label='RMSE')
    ax.scatter([x_min, x_max], y * 2, color='C3', s=25, label='Min, Max')

    ax.set_xlim([x_min - 200, x_max + 200])
    ax.set_ylim([-2, 2])
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('{} Requests Range'.format(request_type))
    ax.set_title('{}-Sample Evaluation Metrics'.format(sample_type))
    ax.legend()
    return fig


def extend_axis_limit(ax_limits):
    '''Extend the matplotlib axis limit by specified percentage.

    Parameters
    ----------
    ax_limits : tuple of floats
        Existing minimum and maximum values of plot axis.

    Returns
    -------
    ax_min : float
        Extended minimum value of plot axis.
    ax_max : float
        Extended maximum value of plot axis.
    '''
    ax_min, ax_max = ax_limits
    return ax_min - (abs(ax_min) * 0.2), ax_max + (ax_max * 0.2)


def plot_forecast_errors(forecast, ci, s_test, alpha_mpl, mae):
    '''Plot errors from in-sample forecast.

    Parameters
    ----------
    forecast : pd.Series
        Out-of-sample forecast values.
    ci : pd.DataFrame
        Out-of-sample forecast confidence intervals.
    s_test : pd.Series
        Series of test set.
    alpha_mpl : float
        Alpha parameter for matplotlib plot.
    mae : float
        Mean Absolute Error of fitted model.

    Returns
    -------
    fig : matplotlib figure
        Plot of MAE and RMSE against min and max of dataset.
    '''
    fore_ci = [forecast.to_frame('forecast'), ci]
    df_fore_ci = (pd.concat(fore_ci, axis='columns'))
    df_fore_ci.rename(
        {'lower count': 'ci_lower', 'upper count': 'ci_upper'},
        axis='columns',
        inplace=True,
    )
    df_fore_ci['fore_error'] = (df_fore_ci['forecast'] - s_test)
    df_fore_ci['fore_error_percent'] = (
            (df_fore_ci['forecast'] / s_test) * 100
    )

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.5)

    ax0.plot(
        df_fore_ci.index,
        df_fore_ci['fore_error'],
        color='C0',
        alpha=alpha_mpl,
        label='Forecast Error',
    )
    ax0.axhline(0, linewidth=1, color='C3', alpha=0.5, label=None)
    ax0.axhline(
        mae,
        linewidth=1,
        linestyle='--',
        color='C1',
        alpha=0.5,
        label='MAE',
    )

    ax0.xaxis.set_major_locator(mdates.MonthLocator())
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    y_min, y_max = extend_axis_limit(ax0.get_ylim())
    ax0.set_ylim(y_min, y_max)
    ax0.set_xlabel('Month')
    ax0.set_ylabel('Request Error')
    ax0.set_title('Out-of-Sample Forecast Errors')
    ax0.legend()

    ax1.plot(
        s_test.index,
        s_test,
        color='C2',
        alpha=alpha_mpl,
        label='Requests',
    )
    ax1.plot(
        df_fore_ci.index,
        df_fore_ci['fore_error'],
        color='C0',
        alpha=alpha_mpl,
        label='Forecast Error',
    )
    ax1.axhline(0, linewidth=1, color='C3', alpha=0.5, label=None)
    ax1.axhline(
        mae,
        linewidth=1,
        linestyle='--',
        color='C1',
        alpha=0.5,
        label='MAE',
    )

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    y_min, y_max = extend_axis_limit(ax1.get_ylim())
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Requests')
    ax1.set_title('Observed Requests vs Out-of-Sample Forecast Errors')
    ax1.legend()

    ax2.plot(
        df_fore_ci.index,
        df_fore_ci['fore_error_percent'],
        color='C0',
        alpha=alpha_mpl,
        label='Forecast Error Percent',
    )
    ax2.axhline(
        100,
        linewidth=1,
        color='C3',
        alpha=0.5,
        label=None,
    )

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    y_min, y_max = extend_axis_limit(ax2.get_ylim())
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Percent')
    ax2.set_title('Out-of-Sample Forecast Errors Percent')
    ax2.legend()

    return fig


def plot_forecast_ci(ci_out):
    '''Plot forecast confidence intervals.

    Parameters
    ----------
    ci_out : pd.Series of floats
        Out-of-sample confidence intervals.

    Returns
    -------
    fig : matplotlib figure
        Plot of MAE and RMSE against min and max of dataset.
    '''

    y_err = (ci_out.iloc[:, 1] - ci_out.iloc[:, 0]) / 2
    x = ci_out.index
    ci_upper = y_err
    ci_lower = -y_err

    y_lim = int(math.ceil((ci_upper.max() / 100.0) * 1.5)) * 100

    fig, ax = plt.subplots(figsize=(12, 3))

    ax.fill_between(x, ci_upper, color='C0', alpha=0.15)
    ax.fill_between(x, ci_lower, color='C0', alpha=0.15)

    ax.plot(ci_upper, color='C0')
    ax.plot(ci_lower, color='C0')
    ax.axhline(linewidth=1, color='C3', alpha=0.5)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel('Month')
    ax.set_ylabel('CI Width')
    ax.set_title('Out-of-Sample Forecast Confidence Intervals')
    return fig


def plot_forecast_volume(model, df_test, alpha_mpl, mae):
    '''Plot forecasted volume of requests by month.

    Parameters
    ----------
    model : statsmodels SARIMAX object
        Object containing fitted SARIMAX results.
    df_test : pd.DataFrame
        Data Frame of test set with datetime index and aggregated
        requests by month.
    alpha_mpl : float
        Alpha parameter for matplotlib plot.
    mae : float
        Mean Absolute Error of fitted model.

    Returns
    -------
    fig0 : matplotlib figure
        Plot unadjusted expected volume by month.
    fig1 : matplotlib figure
        Plot adjusted expected volume by month.
    '''
    pred_out = model.get_forecast(24, dynamic=True)
    forecast_out = pred_out.predicted_mean
    ci_out = pred_out.conf_int()

    x = forecast_out.index
    y = forecast_out
    y_ci = (ci_out.iloc[:, 1] - ci_out.iloc[:, 0]) / 2
    y_bias = y - mae

    fig0, ax = plt.subplots(figsize=(12, 5))

    # Set bar edge color to mitigate overlapping bars due to number
    # of days in a given month
    ax.bar(
        x,
        height=y,
        width=1,
        edgecolor='white',
        yerr=y_ci,
        ecolor='C3',
        alpha=alpha_mpl,
        label='Forecast Requests',
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.set_xlabel('Month')
    ax.set_ylabel('Requests')
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Forecast Volume by Month with Confidence Intervals')
    ax.legend(loc='upper left')

    fig1, ax = plt.subplots(figsize=(12, 3))

    # Set bar edge color to mitigate overlapping bars due to number
    # of days in a given month
    ax.bar(
        x,
        height=y,
        width=1,
        edgecolor='white',
        color='C0',
        alpha=alpha_mpl,
        label='Forecast Requests',
    )
    ax.plot(
        x,
        y_bias,
        color='C1',
        alpha=alpha_mpl,
        label='MAE Adjusted Forecast',
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.set_ylim(0, 8000)
    ax.set_xlabel('Month')
    ax.set_ylabel('Requests')
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Forecast Volume by Month with MAE Adjustment')
    ax.legend(loc='upper left')
    return fig0, fig1


def plot_forecast_values(model, alpha_mpl):
    '''Plot forecasted values of requests by month.

    Parameters
    ----------
    model : statsmodels SARIMAX object
        Object containing fitted SARIMAX results.
    alpha_mpl : float
        Alpha parameter for matplotlib plot.

    Returns
    -------
    fig : matplotlib figure
        Plot forecast expected values by month.
    '''
    pred_out = model.get_forecast(24, dynamic=True)
    forecast_out = pred_out.predicted_mean

    x_pred = forecast_out.loc['2018-11-01':]
    y_pred = forecast_out.loc['2018-11-01':].index

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hlines(y_pred, xmin=0, xmax=x_pred, colors='C0', linewidth=2)
    ax.scatter(x_pred, y_pred)

    for val, idx in zip(x_pred, y_pred):
        ax.text(val + 60, idx, str(int(round(val, 0))), color='C2', va='center')

    ax.yaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.invert_yaxis()
    ax.set_xlabel('Requests')
    ax.set_ylabel('Month')
    ax.set_title('Forecast Volume for Nov 2018 - Oct 2019')
    return fig