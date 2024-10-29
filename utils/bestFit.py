import numpy as np
import scipy.stats as st
def best_fit_distribution(data, bins=300):
    """
    Find the best fit distribution for the given data.

    Parameters
    ----------
    data : array_like
        The data to be fit.
    bins : int
        The number of bins to use for the histogram.

    Returns
    -------
    The best fit distribution and its parameters.
    """

    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0  # Finding the bin center

    # List of distributions to check
    DISTRIBUTIONS = [
        st.alpha, st.beta, st.chi, st.chi2, st.dgamma, st.dweibull, st.erlang,
        st.exponweib, st.f, st.genexpon, st.gausshyper, st.gamma, st.johnsonsb,
        st.johnsonsu, st.norm, st.rayleigh, st.rice, st.recipinvgauss, st.t,
        st.weibull_min, st.weibull_max
    ]

    best_distribution = st.norm
    best_sse = np.inf
    best_params = (0.0, 1.0)

    # Iterate through all distributions and find the one with the lowest SSE
    for distribution in DISTRIBUTIONS:
        params = distribution.fit(data)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Calculate the PDF of the distribution
        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        # Calculate the SSE between the observed histogram and the PDF
        sse = np.sum(np.power(y - pdf, 2.0))

        # Update the best distribution if the SSE is lower
        if best_sse > sse > 0:
            best_sse = sse
            best_distribution = distribution
            best_params = params

    return best_distribution, best_params
