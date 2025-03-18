import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity


class PDFEstimator:
    """
    A class that provides methods to estimate probability density functions
    from sample data and evaluate p(x) at any point.
    """

    def __init__(self, data, method="kde", bandwidth=None, bins=50, kernel="gaussian"):
        """
        Initialize the PDF estimator with sample data.

        Parameters:
        -----------
        data : array-like
            Sample data to estimate PDF from
        method : str
            Method to use: 'kde'
        bandwidth : float or str or None
            Bandwidth for KDE methods (None for automatic selection)
            For scipy_kde, can also be 'scott' or 'silverman'
        bins : int
            Number of bins for histogram method
        kernel : str
            Kernel type for sklearn_kde ('gaussian', 'tophat', etc.)
        """
        self.data = np.asarray(data).flatten()
        self.method = method
        self.bandwidth = bandwidth
        self.bins = bins
        self.kernel = kernel

        # Create PDF estimator based on chosen method
        if method == "kde":
            self._create_scipy_kde()
        else:
            raise ValueError(f"Method '{method}' not recognized. Use 'kde'")

    def _create_scipy_kde(self):
        """Create PDF estimator using scipy's gaussian_kde."""
        self.kde = stats.gaussian_kde(self.data, bw_method=self.bandwidth)

        # Define pdf function
        self.pdf = lambda x: self.kde(x)

    def evaluate(self, x):
        """
        Evaluate the PDF at point(s) x.

        Parameters:
        -----------
        x : float or array-like
            Point(s) to evaluate the PDF at

        Returns:
        --------
        float or ndarray
            PDF value(s) at point(s) x
        """
        return self.pdf(x)
