import numpy as np
from scipy.interpolate import interp1d

def inverse_transform_sampling(data, bins, nsamples):
    """
    Extract random samples from a distribution
    using inverse transform sampling. This code
    uses a Univariate Spline (without smoothing)
    to extract the inverse CDF for an arbitrary
    continuous distribution.

    Parameters
    ------------------
    data : array_like
        1D distribution to be sampled.
    bins : int or array_like
        Histogram bins for `data` (see np.histogram).
    nsamples : int
        Number of samples (>1) to be returned.

    Returns
    --------------------
    samples : np.array
        Samples according to the distribution of `data`.
        Length matches nsamples.
    """
    hist, binedges = np.histogram(data, bins=bins, density=True)
    cum_values = np.zeros_like(binedges)
    cum_values[1:] = np.cumsum(hist*(binedges[1:]-binedges[:-1]))
    inversecdf = interp1d(cum_values, binedges)
    uval = np.random.rand(nsamples)
    samples = inversecdf(uval)
    return samples



if __name__=='__main__':
    data = np.random.normal(loc=140,scale=26,size=1000)
    samples = inverse_transform_sampling(data, bins=40, nsamples=500)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(data, bins=40, label='Raw Data')
    plt.hist(samples, histtype='step', color='orange', linewidth=3, label='Samples')
    plt.legend(loc='best')
    plt.show() 
