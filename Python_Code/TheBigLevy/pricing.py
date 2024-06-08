# import packages
import numpy as np
from scipy.stats import norm, invgauss, gamma
from generals import yearfrac_act_365
from scipy.interpolate import interp1d
from scipy.integrate import quad
import pandas as pd

# Function to compute interpolated discounts
def int_ext_df(discounts, setDate, dates, target_date):
    """
    Interpolates (linear) or extrapolates (flat) the zero rates curve for a given date.

    Parameters:
    discounts : array_like
        Discount factors curve for preceding values (Bootstrap).
    setDate : array_like
        Settlement Date
    dates : array_like
        Dates of discount factors curve bootstrap.
    target_date : array_like
        Corresponding dates of the discount requested.

    Returns:
    array_like
        Discount factors found by matching/interpolating/extrapolating the zero rates curve.
    """

    year_fractions = yearfrac_act_365(setDate, dates)
    jolly_years = target_date - setDate
    jolly_days = jolly_years.days
    target_year_fractions = np.array(jolly_days/365)

    # Compute the zero rates
    zero_rates = -np.log(discounts) / year_fractions

    # Interpolate the zero rates
    interp_func = interp1d(year_fractions, zero_rates, kind='linear', fill_value='extrapolate')
    search_rates = interp_func(target_year_fractions)

    # Reconstruct the discount factors
    reconstructed_discounts = np.exp(-search_rates * target_year_fractions)

    return reconstructed_discounts


# Function to compute derivative price using Black
def black_pricing(Market_US, Market_EU, setDate, targetDate, MeanBMs, rho, N_sim):
    """
    This function computes the price of a derivative with the following payoff:
    Payoff = max(S1(t) - S1(0), 0) * I(S2(t) < 0.95 * S2(0))

    Parameters:
    Market_US : dict
        Market data of the US asset.
    Market_EU : dict
        Market data of the European asset.
    setDate : datetime
        Date of the simulation.
    targetDate : datetime
        Target date of the simulation.
    MeanBMs : array_like
        Mean of the Brownian motions.
    rho : float
        Correlation between the Brownian motions.
    N_sim : int
        Number of simulations.

    Returns:
    tuple
        price and priceCI.
    """
    # Recall variables from the market data
    S0_US = Market_US.spot
    S0_EU = Market_EU.spot
    Expiries_US = Market_US.datesExpiry
    Expiries_EU = Market_EU.datesExpiry
    B_bar_US = Market_US.B_bar
    B_bar_EU = Market_EU.B_bar
    sigma_US = Market_US.sigma
    sigma_EU = Market_EU.sigma

    # Compute the discount via interpolation
    discount_US = int_ext_df(B_bar_US,setDate,Expiries_US, targetDate)
    discount_EU = int_ext_df(B_bar_EU,setDate, Expiries_EU, targetDate)

    # Compute the forward prices
    F0_US = S0_US / discount_US
    F0_EU = S0_EU / discount_EU

    # Define the Covariance matrix
    cov_matrix = np.array([[1, rho], [rho, 1]])

    # Generate correlated random variables from the multivariate normal distribution
    Z = np.random.multivariate_normal(MeanBMs, cov_matrix, N_sim)

    # Compute the time to maturity
    ttm = (targetDate - setDate).days / 365.0

    # Simulate the assets via GBM (Geometric Brownian Motion)
    S1_US = F0_US * np.exp((-0.5 * sigma_US ** 2) * ttm + sigma_US * np.sqrt(ttm) * Z[:, 0])
    S1_EU = F0_EU * np.exp((-0.5 * sigma_EU ** 2) * ttm + sigma_EU * np.sqrt(ttm) * Z[:, 1])

    # Compute the payoff
    Indicator_function = (S1_EU < 0.95 * S0_EU)
    payoff = np.maximum(S1_US - S0_US, 0) * Indicator_function

    # Compute the price
    price = discount_US * np.mean(payoff)

    # Confidence interval
    a = 0.01
    CI = norm.ppf(1 - a) * np.std(payoff) / np.sqrt(N_sim)
    priceCI = (price - CI, price + CI)

    return price, priceCI


# Function to compute derivative price using closed formula
def closedFormula(Market_US, Market_EU, set_date, target_date, rho):
    """
    This function computes the price of a derivative with payoff:
    Payoff = max(S1(t) - S1(0), 0) * I(S2(t) < 0.95 * S2(0))
    with a closed formula (the integral is numerically computed).

    Parameters:
    Market_US : dict
        US market data (dictionary)
    Market_EU : dict
        EU market data (dictionary)
    set_date : datetime
        Settlement date
    target_date : datetime
        Maturity of the derivative
    rho : float
        Correlation between the markets

    Returns:
    float
        Price of the derivative using closed formula
    """
    # Import spot prices
    S0_US = Market_US.spot
    # S0_EU = Market_EU.spot

    # Import expiries
    Expiries_US = Market_US.datesExpiry
    Expiries_EU = Market_EU.datesExpiry

    # Import discount factors
    B_bar_US = Market_US.B_bar
    B_bar_EU = Market_EU.B_bar

    # Import volatilities
    sigma_US = Market_US.sigma
    sigma_EU = Market_EU.sigma

    # Compute discounts
    discount_US = int_ext_df(B_bar_US, set_date, Expiries_US, target_date)
    discount_EU = int_ext_df(B_bar_EU, set_date, Expiries_EU, target_date)

    # Time to maturity
    ttm = (target_date - set_date).days / 365.0

    # Zero rates for EU & US
    zero_rate_US = -np.log(discount_US) / ttm
    zero_rate_EU = -np.log(discount_EU) / ttm

    # Boundary for the integral
    x_min = -np.inf
    x_max = (np.log(0.95) - (zero_rate_EU - 0.5 * sigma_EU ** 2) * ttm) / sigma_EU

    def A(omega):
        return ((zero_rate_US - 0.5 * sigma_US ** 2) * ttm + sigma_US * (
                    ttm * (1 - rho ** 2) * sigma_US + rho * omega)) / (sigma_US * np.sqrt(ttm * (1 - rho ** 2)))

    def B(omega):
        return A(omega) - np.sqrt(ttm * (1 - rho ** 2)) * sigma_US

    # Function to integrate
    def integrand(omega):
        term1 = np.exp(zero_rate_US * ttm - 0.5 * sigma_US ** 2 * rho ** 2 * ttm + sigma_US * rho * omega) * norm.cdf(
            A(omega))
        term2 = norm.cdf(B(omega))
        return (term1 - term2) * 1 / np.sqrt(2 * np.pi * ttm) * np.exp(-0.5 * omega ** 2 / ttm)

    # Compute the integral
    I, _ = quad(integrand, x_min, x_max)
    # Price closed formula
    price_closed_formula = discount_US * S0_US * I

    return price_closed_formula

# Function to compute derivative price using closed formula
def levy_pricing2(Market_US, Market_EU, settlement, target_date, kappa_US, kappa_EU, sigma_US, sigma_EU, theta_US,
                  theta_EU, rho, N_sim, flag):
    """
    Computes the price of a derivative using a Monte Carlo simulation under Lévy processes.

    Parameters:
    Market_US : dict
        Market data for the US market.
    Market_EU : dict
        Market data for the EU market.
    settlement : datetime
        Settlement date.
    target_date : datetime
        Target date.
    kappa_US : float
        Lévy parameter for the US market.
    kappa_EU : float
        Lévy parameter for the EU market.
    sigma_US : float
        Volatility for the US market.
    sigma_EU : float
        Volatility for the EU market.
    theta_US : float
        Drift for the US market.
    theta_EU : float
        Drift for the EU market.
    rho : float
        Correlation between the markets.
    N_sim : int
        Number of simulations.
    flag : str
        Flag to choose the model ('NIG' or 'Error').

    Returns:
    tuple
        Price of the derivative and confidence interval.
    """
    # Recall variables from the market data
    S0_US = Market_US.spot
    S0_EU = Market_EU.spot

    # Import the expiries
    Expiries_US = Market_US.datesExpiry
    Expiries_EU = Market_EU.datesExpiry

    # Import the market discounts B_bar
    B_bar_US = Market_US.B_bar
    B_bar_EU = Market_EU.B_bar

    # Compute the discount via interpolation
    discount_US = int_ext_df(B_bar_US, settlement, Expiries_US, target_date)
    discount_EU = int_ext_df(B_bar_EU, settlement, Expiries_EU, target_date)

    # Compute the forward prices
    F0_US = S0_US / discount_US
    F0_EU = S0_EU / discount_EU

    # Compute the time to maturity
    ttm = (target_date - settlement).days / 365.0

    # Draw the standard normal random variables
    g = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], N_sim)

    if flag == 'NIG':
        # Draw the inverse Gaussian random variables
        G_EU = invgauss.rvs(mu=ttm / kappa_EU, scale=1, size=N_sim)
        G_US = invgauss.rvs(mu=ttm / kappa_US, scale=1, size=N_sim)

        # NIG model
        compensatorEU_NIG = -ttm / kappa_EU * (1 - np.sqrt(1 - 2 * kappa_EU * theta_EU - kappa_EU * sigma_EU ** 2))
        compensatorUS_NIG = -ttm / kappa_US * (1 - np.sqrt(1 - 2 * kappa_US * theta_US - kappa_US * sigma_US ** 2))

        ft_EU = np.sqrt(ttm) * sigma_EU * np.sqrt(G_EU) * g[:, 0] - (
                    0.5 + theta_EU) * ttm * sigma_EU ** 2 * G_EU + compensatorEU_NIG
        ft_US = np.sqrt(ttm) * sigma_US * np.sqrt(G_US) * g[:, 1] - (
                    0.5 + theta_US) * ttm * sigma_US ** 2 * G_US + compensatorUS_NIG

    else:
        raise ValueError('Flag not recognized')

        # Simulate asset prices
    S1_US = F0_US * np.exp(ft_US)
    S1_EU = F0_EU * np.exp(ft_EU)

    # Indicator function
    Indicator_function = (S1_EU < 0.95 * S0_EU)

    # Compute the payoff
    payoff = np.maximum(S1_US - S0_US, 0) * Indicator_function

    # Compute the price of the derivative
    price = discount_US * np.mean(payoff)

    # Confidence interval
    a = 0.01
    CI = norm.ppf(1 - a) * np.std(payoff) / np.sqrt(N_sim)
    priceCI = [price - CI, price + CI]

    return price, priceCI
