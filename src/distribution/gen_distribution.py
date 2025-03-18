import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class DistributionGenerator:
    """
    A class to generate and visualize common probability distributions.
    """

    def __init__(self, random_seed=None):
        """
        Initialize the distribution generator.

        Parameters:
        -----------
        random_seed : int, optional
            Seed for random number generation for reproducibility.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

    def normal(self, mean=0, std=1, size=1000):
        """
        Generate samples from a normal (Gaussian) distribution.

        Parameters:
        -----------
        mean : float
            Mean (μ) of the distribution - controls the center of the bell curve.
            Range: Any real number (-∞ to +∞)
        std : float
            Standard deviation (σ) of the distribution - controls the spread/width of the bell curve.
            Range: std > 0
        size : int
            Number of samples to generate.
            Range: size > 0

        Returns:
        --------
        numpy.ndarray
            Array of samples from the normal distribution.
        """
        return np.random.normal(mean, std, size)

    def uniform(self, low=0, high=1, size=1000):
        """
        Generate samples from a uniform distribution.

        Parameters:
        -----------
        low : float
            Lower bound of the distribution - minimum value in the range.
            Range: Any real number, must be less than high
        high : float
            Upper bound of the distribution - maximum value in the range.
            Range: Any real number, must be greater than low
        size : int
            Number of samples to generate.
            Range: size > 0

        Returns:
        --------
        numpy.ndarray
            Array of samples from the uniform distribution.
        """
        return np.random.uniform(low, high, size)

    def exponential(self, scale=1.0, size=1000):
        """
        Generate samples from an exponential distribution.

        Parameters:
        -----------
        scale : float
            Scale parameter (1/λ) of the distribution - controls the rate of decay.
            Larger values stretch out the distribution.
            Range: scale > 0
        size : int
            Number of samples to generate.
            Range: size > 0

        Returns:
        --------
        numpy.ndarray
            Array of samples from the exponential distribution.
        """
        return np.random.exponential(scale, size)

    def poisson(self, lam=1.0, size=1000):
        """
        Generate samples from a Poisson distribution.

        Parameters:
        -----------
        lam : float
            Rate parameter (λ) of the distribution - controls both mean and variance.
            Higher values increase the average number of events.
            Range: lam > 0
        size : int
            Number of samples to generate.
            Range: size > 0

        Returns:
        --------
        numpy.ndarray
            Array of samples from the Poisson distribution.
        """
        return np.random.poisson(lam, size)

    def binomial(self, n=10, p=0.5, size=1000):
        """
        Generate samples from a binomial distribution.

        Parameters:
        -----------
        n : int
            Number of trials - controls the maximum possible value.
            Range: n > 0
        p : float
            Probability of success in each trial - controls the skewness.
            Range: 0 ≤ p ≤ 1
        size : int
            Number of samples to generate.
            Range: size > 0

        Returns:
        --------
        numpy.ndarray
            Array of samples from the binomial distribution.
        """
        return np.random.binomial(n, p, size)

    def gamma(self, shape=1.0, scale=1.0, size=1000):
        """
        Generate samples from a gamma distribution.

        Parameters:
        -----------
        shape : float
            Shape parameter (k) of the distribution - controls the basic shape.
            Lower values make it more skewed, higher values make it more symmetric.
            Range: shape > 0
        scale : float
            Scale parameter (θ) of the distribution - controls the spread.
            Larger values stretch out the distribution.
            Range: scale > 0
        size : int
            Number of samples to generate.
            Range: size > 0

        Returns:
        --------
        numpy.ndarray
            Array of samples from the gamma distribution.
        """
        return np.random.gamma(shape, scale, size)

    def beta(self, a=1.0, b=1.0, size=1000):
        """
        Generate samples from a beta distribution.

        Parameters:
        -----------
        a : float
            Alpha parameter - controls the shape of the left tail.
            Higher values shift the distribution right.
            Range: a > 0
        b : float
            Beta parameter - controls the shape of the right tail.
            Higher values shift the distribution left.
            Range: b > 0
        size : int
            Number of samples to generate.
            Range: size > 0

        Returns:
        --------
        numpy.ndarray
            Array of samples from the beta distribution.
        """
        return np.random.beta(a, b, size)

    def lognormal(self, mean=0, sigma=1, size=1000):
        """
        Generate samples from a log-normal distribution.

        Parameters:
        -----------
        mean : float
            Mean of the underlying normal distribution - controls the location.
            Range: Any real number (-∞ to +∞)
        sigma : float
            Standard deviation of the underlying normal distribution - controls the spread.
            Larger values increase right skewness.
            Range: sigma > 0
        size : int
            Number of samples to generate.
            Range: size > 0

        Returns:
        --------
        numpy.ndarray
            Array of samples from the log-normal distribution.
        """
        return np.random.lognormal(mean, sigma, size)

    def chi_square(self, df=1, size=1000):
        """
        Generate samples from a chi-square distribution.

        Parameters:
        -----------
        df : int
            Degrees of freedom - controls the shape and location.
            Higher values make the distribution more symmetric and Gaussian-like.
            Range: df > 0
        size : int
            Number of samples to generate.
            Range: size > 0

        Returns:
        --------
        numpy.ndarray
            Array of samples from the chi-square distribution.
        """
        return np.random.chisquare(df, size)

    def student_t(self, df=1, size=1000):
        """
        Generate samples from a Student's t-distribution.

        Parameters:
        -----------
        df : float
            Degrees of freedom - controls the heaviness of the tails.
            Lower values give heavier tails, higher values approach normal distribution.
            Range: df > 0
        size : int
            Number of samples to generate.
            Range: size > 0

        Returns:
        --------
        numpy.ndarray
            Array of samples from the Student's t-distribution.
        """
        return np.random.standard_t(df, size)
