import numpy as np
from typing import Callable, List, Dict, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns


class InformationTheory:
    """
    A class for calculating and using core information theory metrics:
    - Shannon Entropy: Measures uncertainty or randomness in a distribution
    - Cross-Entropy: Measures the average number of bits needed when using one distribution
                     to encode samples from another
    - KL Divergence: Measures information loss when approximating one distribution with another
    """

    def __init__(self, epsilon=1e-10, base=2):
        """
        Initialize the Information Theory calculator.

        Parameters:
        -----------
        epsilon : float
            A small value to avoid log(0) errors
        base : int or float
            Base for logarithm (2 for bits, e for nats, 10 for dits)
        """
        self.epsilon = epsilon
        self.base = base

        # Set log function based on the base
        if base == 2:
            self.log_func = np.log2
            self.units = "bits"
        elif base == np.e:
            self.log_func = np.log
            self.units = "nats"
        elif base == 10:
            self.log_func = np.log10
            self.units = "dits"
        else:
            self.log_func = lambda x: np.log(x) / np.log(base)
            self.units = f"units (base {base})"

    def get_entropy(self, pdf, x_values):
        """
        Calculate the Shannon entropy of a probability distribution.

        H(P) = -∑ P(x) * log(P(x))

        Conceptually, this represents the average amount of "surprise" in
        observing samples from the distribution, or the average number of
        bits needed to encode samples from P.

        Parameters:
        -----------
        pdf : callable
            Probability density/mass function that can be evaluated at points
        x_values : array-like
            The values of the random variable

        Returns:
        --------
        float
            The entropy value in bits (or specified base units)
        """
        # Calculate probabilities for each x value
        probabilities = np.array([pdf.evaluate(x) for x in x_values])

        # Ensure probabilities sum to 1
        probabilities = probabilities / np.sum(probabilities)

        # Add epsilon to avoid log(0)
        safe_probs = probabilities + self.epsilon

        # Calculate entropy
        entropy = -np.sum(safe_probs * self.log_func(safe_probs))
        return entropy

    def get_cross_entropy(self, pdf_p, pdf_q, x_values):
        """
        Calculate the cross-entropy between two probability distributions.

        H(P, Q) = -∑ P(x) * log(Q(x))

        Interpretation: The average number of bits needed to encode events from
        distribution P when using an optimal code for distribution Q.

        Always H(P, Q) ≥ H(P), with equality only when P = Q.

        Parameters:
        -----------
        pdf_p : callable
            First probability density/mass function (true distribution)
        pdf_q : callable
            Second probability density/mass function (approximation)
        x_values : array-like
            The values of the random variable

        Returns:
        --------
        float
            The cross-entropy value in bits (or specified base units)
        """
        # Calculate probabilities for each distribution
        p_probs = np.array([pdf_p.evaluate(x) for x in x_values])
        q_probs = np.array([pdf_q.evaluate(x) for x in x_values])

        # Normalize probabilities
        p_probs = p_probs / np.sum(p_probs)
        q_probs = q_probs / np.sum(q_probs)

        # Add epsilon to avoid log(0)
        safe_q_probs = q_probs + self.epsilon

        # Calculate cross-entropy
        cross_entropy = -np.sum(p_probs * self.log_func(safe_q_probs))
        return cross_entropy

    def get_kl_divergence(self, pdf_p, pdf_q, x_values):
        """
        Calculate the Kullback-Leibler divergence from P to Q.

        KL(P||Q) = ∑ P(x) * log(P(x)/Q(x))

        Interpretation: The information lost when Q is used to approximate P.
        KL divergence is always non-negative and is zero only when P = Q.
        It is not symmetric: KL(P||Q) ≠ KL(Q||P) in general.

        Parameters:
        -----------
        pdf_p : callable
            First probability density/mass function (true distribution)
        pdf_q : callable
            Second probability density/mass function (approximation)
        x_values : array-like
            The values of the random variable

        Returns:
        --------
        float
            The KL divergence value in bits (or specified base units)
        """
        # Calculate probabilities for each distribution
        p_probs = np.array([pdf_p.evaluate(x) for x in x_values])
        q_probs = np.array([pdf_q.evaluate(x) for x in x_values])

        # Normalize probabilities
        p_probs = p_probs / np.sum(p_probs)
        q_probs = q_probs / np.sum(q_probs)

        # Add epsilon to avoid division by zero or log(0)
        safe_p_probs = p_probs + self.epsilon
        safe_q_probs = q_probs + self.epsilon

        # Calculate KL divergence
        kl_div = np.sum(safe_p_probs * self.log_func(safe_p_probs / safe_q_probs))
        return kl_div

    def get_kl_divergence_from_entropy(self, pdf_p, pdf_q, x_values):
        """
        Calculate KL divergence using the relationship:
        KL(P||Q) = H(P,Q) - H(P)

        This demonstrates the key relationship between entropy, cross-entropy,
        and KL divergence.

        Parameters:
        -----------
        pdf_p : callable
            First probability density/mass function (true distribution)
        pdf_q : callable
            Second probability density/mass function (approximation)
        x_values : array-like
            The values of the random variable

        Returns:
        --------
        float
            The KL divergence value in bits (or specified base units)
        """
        entropy_p = self.get_entropy(pdf_p, x_values)
        cross_entropy = self.get_cross_entropy(pdf_p, pdf_q, x_values)
        return cross_entropy - entropy_p

    def get_pointwise_kl(self, pdf_p, pdf_q, x_values):
        """
        Calculate the pointwise KL divergence contributions.

        This is useful for visualizing which parts of the distributions
        contribute most to the total KL divergence.

        Parameters:
        -----------
        pdf_p : callable
            First probability density/mass function
        pdf_q : callable
            Second probability density/mass function
        x_values : array-like
            The values of the random variable

        Returns:
        --------
        np.ndarray
            Pointwise KL divergence contributions at each x
        float
            Total KL divergence (sum of pointwise contributions)
        """
        # Calculate probabilities for each distribution
        p_probs = np.array([pdf_p.evaluate(x) for x in x_values])
        q_probs = np.array([pdf_q.evaluate(x) for x in x_values])

        # Normalize probabilities
        p_probs = p_probs / np.sum(p_probs)
        q_probs = q_probs / np.sum(q_probs)

        # Add epsilon to avoid division by zero or log(0)
        safe_p_probs = p_probs + self.epsilon
        safe_q_probs = q_probs + self.epsilon

        # Calculate pointwise KL contributions
        ratio = safe_p_probs / safe_q_probs
        log_ratio = self.log_func(ratio)
        pointwise_kl = safe_p_probs * log_ratio

        # Total KL divergence
        total_kl = np.sum(pointwise_kl)

        return pointwise_kl, total_kl

    def compare_forward_reverse_kl(
        self, pdf_p, pdf_q, x_values, p_name="P", q_name="Q"
    ):
        """
        Compare KL(P||Q) and KL(Q||P) to demonstrate asymmetry.

        Parameters:
        -----------
        pdf_p, pdf_q : callable
            Probability density/mass functions
        x_values : array-like
            The values of the random variable
        p_name, q_name : str
            Names of the distributions

        Returns:
        --------
        tuple
            (KL(P||Q), KL(Q||P), difference)
        """
        # Calculate forward KL: KL(P||Q)
        forward_kl = self.get_kl_divergence(pdf_p, pdf_q, x_values)

        # Calculate reverse KL: KL(Q||P)
        reverse_kl = self.get_kl_divergence(pdf_q, pdf_p, x_values)

        # Calculate difference
        difference = abs(forward_kl - reverse_kl)

        # Print results
        print(f"KL({p_name}||{q_name}) = {forward_kl:.4f} {self.units}")
        print(f"KL({q_name}||{p_name}) = {reverse_kl:.4f} {self.units}")
        print(f"Difference: {difference:.4f} {self.units}")

        return forward_kl, reverse_kl, difference
