import numpy as np
import scipy.optimize as op
import seaborn as sns

def four_param_logistic(p):
    """4p logistic function maker.
    
    Returns a function that accepts x and returns y for
    the 4-parameter logistic defined by p.
    
    The 4p logistic is defined by:
    y = A + (K - A) / (1 + exp(-B*(x-M)))
    
    Args:
        p: an iterable of length 4
            A, K, B, M = p
    
    Returns:
        A function that accepts a numpy array as an argument 
        for x values and returns the y values for the defined 4pl curve.
            
    Marvin Thielk 2016
    mthielk@ucsd.edu
    """
    A, K, B, M = p
    def f(x):
        return A + (K - A) / (1 + np.exp(-B*(x-M)))
    return f

def ln_like(p, x, y):
    """log likelihood for fitting the four parameter logistic.
    
    Args:
        p: an iterable of length 4
            A, K, B, M = p
        x: a numpy array of length n
        y: a numpy array of length n
            must be of dtype double or float so multiplication works
    
    Returns:
        The log-likelihood that the samples y are drawn from a distribution
        where the 4pl(x; p) is the probability of getting y=1
            
    Marvin Thielk 2016
    mthielk@ucsd.edu
    """
    p_4pl = four_param_logistic(p)
    probs = p_4pl(x)
    return np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))

def dln_like(p, x, y):
    """gradient of the log likelihood for fitting the four parameter logistic.
    
    Args:
        p: an iterable of length 4
            A, K, B, M = p
        x: a numpy array of length n
        y: a numpy array of length n
            must be of dtype double or float so multiplication works
    
    Returns:
        The gradient of the log-likelihood that the samples y are drawn from 
        a distribution where the 4pl(x; p) is the probability of getting y=1
            
    Marvin Thielk 2016
    mthielk@ucsd.edu
    """
    A, K, B, M = p
    def f(x):
        return A + (K - A) / (1 + np.exp(-B*(x-M)))
    def df(x):
        temp1 = np.exp(-B*(x-M))
        dK = 1. / (1. + temp1)
        dA = 1. - dK
        temp2 = temp1 / (1. + temp1) ** 2
        dB = (K - A) * (x - M) * temp2
        dM = -(K - A) * B * temp2
        return np.vstack((dA, dK, dB, dM))
    p_4pl = f(x)
    d_p_4pl = df(x)
    return np.sum(y * d_p_4pl / (p_4pl) - (1 - y) * d_p_4pl / (1 - p_4pl), 1)

def nll(*args):
    """negative log-likelihood for fitting the 4 param logistic."""
    return -ln_like(*args)

def ndll(*args):
    """negative grad of the log-likelihood for fitting the 4 param logistic."""
    return -dln_like(*args)

def est_pstart(x, y):
    """basic estimation of a good place to start log likelihood maximization.
    
    Args:
        x: a numpy array of length n
            assumes a finite number of unique x values
        y: a numpy array of length n
            must be of dtype double or float so multiplication works
    
    Returns:
        p_start: an iterable of length 4 that should be a reasonable spot to
            start the optimization
            A, K, B, M = p_start
            
    Marvin Thielk 2016
    mthielk@ucsd.edu
    """
    p_start = [.01, .99, .2, 0]
    x_vals = np.unique(x)
    p_start[3] = np.mean(x_vals)
    y_est = np.array([np.mean(y[x==i]) for i in x_vals])
    midpoint_est = np.mean(np.where((y_est[0:-1]<.5) & (y_est[1:]>=.5)))
    if np.isnan(midpoint_est):
        return p_start
    p_start[3] = midpoint_est
    return p_start

def fit_4pl(x, y, p_start=None, verbose=False):
    """Fits a 4 parameter logistic function to the data.
    
    Args:
        x: a numpy array of length n
            assumes a finite number of unique x values
        y: a numpy array of length n
            must be of dtype double or float so multiplication works
    optional:
        p_start: an iterable of length 4 that would be a reasonable spot to
            start the optimization. If None, tries to estimate it.
            A, K, B, M = p_start
            default=None
        verbose: boolean flag that allows printing of more error messages.
    
    Returns:
        p_result: an iterable of length 4 that defines the model that 
        is maximally likely
            A, K, B, M = p_result
            
    Marvin Thielk 2016
    mthielk@ucsd.edu
    """
    try:
        if not p_start:
            p_start = est_pstart(x, y)
    except TypeError:
        pass
    for i in range(3):
        if verbose and i > 0:
            print 'retry', i
        result = op.minimize(nll, p_start, args=(x, y), jac=ndll, bounds=((eta_bounds,1-eta_bounds), (eta_bounds,1-eta_bounds), (None, None), (None, None)))
        if result.success:
            return result.x
        else:
            if verbose:
                print subj, dim, p_start, 'failure', result
            p_start = result.x
    return False

def plot_4pl(x, y, color=None, **kwargs):
    """helper for lmplot_4pl."""
    data = kwargs.pop("data")
    x = np.array(data[x].tolist())
    y = np.array(data[y].tolist(), dtype=double)

    result = fit_4pl(x, y)
    try:
        result_4pl = four_param_logistic(result)
        t = range(128)
        
        if color is None:
            lines, = plot(x.mean(), y.mean())
            color = lines.get_color()
            lines.remove()
        
        plot(t, result_4pl(t), color=color)
    except TypeError:
        pass
    
def lmplot_4pl(x, y, data, hue=None, col=None, row=None, palette=None,
           col_wrap=None, size=5, aspect=1, markers="o", sharex=True,
           sharey=True, hue_order=None, col_order=None, row_order=None,
           legend=True, legend_out=True, x_estimator=None, x_bins=None,
           x_ci="ci", scatter=True, fit_reg=False, ci=95, n_boot=1000,
           units=None, order=1, logistic=True, lowess=False, robust=False,
           logx=False, x_partial=None, y_partial=None, truncate=False,
           x_jitter=None, y_jitter=None, scatter_kws=None, line_kws=None):
    """lmplot from seaborn modified to plot a 4 parameter logistic
    
    See seaborn documentation for most options
    
    differences:
        set fit_reg=False (default) if you only want to plot 4p-logistic
            and not the standard 2 parameter logistic regression
            
    TODO(Marvin): better package 4pl fit code to be able to just drop in 
        using the built in functions and options
    """

    # Reduce the dataframe to only needed columns
    need_cols = [x, y, hue, col, row, units, x_partial, y_partial]
    cols = np.unique([a for a in need_cols if a is not None]).tolist()
    data = data[cols]

    # Initialize the grid
    facets = sns.FacetGrid(data, row, col, hue, palette=palette,
                       row_order=row_order, col_order=col_order,
                       hue_order=hue_order, size=size, aspect=aspect,
                       col_wrap=col_wrap, sharex=sharex, sharey=sharey,
                       legend_out=legend_out)

    # Add the markers here as FacetGrid has figured out how many levels of the
    # hue variable are needed and we don't want to duplicate that process
    if facets.hue_names is None:
        n_markers = 1
    else:
        n_markers = len(facets.hue_names)
    if not isinstance(markers, list):
        markers = [markers] * n_markers
    if len(markers) != n_markers:
        raise ValueError(("markers must be a singeton or a list of markers "
                          "for each level of the hue variable"))
    facets.hue_kws = {"marker": markers}

    # Hack to set the x limits properly, which needs to happen here
    # because the extent of the regression estimate is determined
    # by the limits of the plot
    if sharex:
        for ax in facets.axes.flat:
            ax.scatter(data[x], np.ones(len(data)) * data[y].mean()).remove()

    # Draw the regression plot on each facet
    regplot_kws = dict(
        x_estimator=x_estimator, x_bins=x_bins, x_ci=x_ci,
        scatter=scatter, fit_reg=fit_reg, ci=ci, n_boot=n_boot, units=units,
        order=order, logistic=logistic, lowess=lowess, robust=robust,
        logx=logx, x_partial=x_partial, y_partial=y_partial, truncate=truncate,
        x_jitter=x_jitter, y_jitter=y_jitter,
        scatter_kws=scatter_kws, line_kws=line_kws,
        )
    facets.map_dataframe(plot_4pl, x, y, **regplot_kws)
    facets.map_dataframe(sns.regplot, x, y, **regplot_kws)

    # Add a legend
    if legend and (hue is not None) and (hue not in [col, row]):
        facets.add_legend()
    return facets
    