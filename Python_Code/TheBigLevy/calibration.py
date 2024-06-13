# import packages
import numpy as np
from scipy.interpolate import interp1d


class Calibrated_OptionMarketData:
    def __init__(self, datesExpiry, strikes, spot, B_bar, F0, sigma, kappa, theta):
        self.datesExpiry = datesExpiry
        self.strikes = strikes
        self.spot = spot
        self.B_bar = B_bar
        self.F0 = F0
        self.sigma = sigma
        self.kappa = kappa
        self.theta = theta


    def __repr__(self):
        return (f"OptionMarketData(datesExpiry={len(self.datesExpiry)}, strikes={len(self.strikes)},"
                f"spot={len(self.spot)}, B_bar={len(self.B_bar)}, F0={len(self.F0)},"
                f"sigma={len(self.sigma)}, kappa={len(self.kappa)}, theta={len(self.theta)}"
                )


# Functions to compute integral (using FFT)
def callIntegral(B0, F0, sigma, kappa, eta, t, log_moneyness, M, dz, flag):
    """
    Compute the price of a call option using the integral of Normal Mean-Variance Mixture model

    INPUT:
    B0: discount factor at time 0
    F0: forward price at time 0
    sigma: variance of the model
    kappa: vol of vol
    eta: skewness
    t: time to maturity
    log_moneyness: log of the moneyness to compute the price at
    M: N = 2^M, number of nodes for the FFT and quadrature
    flag: flag to choose NIG or VG

    OUTPUT:
    callPrices: price of the call option (same size as log_moneyness)
    """

    # Compute the compensator
    compensator_NIG = - t / kappa * (1 - np.sqrt(1 - 2 * kappa * eta - kappa * sigma ** 2))

    # Define the characteristic function
    phi = lambda xi: np.exp(
        t * (1 / kappa * (1 - np.sqrt(1 - 2j * xi * kappa * eta + xi ** 2 * kappa * sigma ** 2)))) \
                     * np.exp(1j * xi * compensator_NIG)


    # Compute the integral via fast fourier transform
    I = integralFFT(phi, M, dz, log_moneyness)

    # Apply the lewis formula
    callPrices = B0 * F0 * (1 - np.exp(-log_moneyness / 2) * I)

    return callPrices


def integralFFT(phi, M, dz, queryPoints):
    """
    This function computes the Fourier Transform of the input integrand
    using the FFT algorithm.

    INPUTS:
    phi (function): Function handle to the integrand
    M (int): N = 2^M number of points to use in the FFT
    dz (float): Grid spacing
    queryPoints (array-like): Points to interpolate the integral values

    OUTPUTS:
    I (array): The integral of the integrand
    """

    # Compute N
    N = 2 ** M

    # Compute the z values
    z_1 = -(N - 1) / 2 * dz
    z = np.arange(z_1, -z_1 + dz, dz)

    # Compute the d_xi value
    d_xi = 2 * np.pi / (N * dz)
    xi_1 = -(N - 1) / 2 * d_xi
    xi = np.arange(xi_1, -xi_1 + d_xi, d_xi)

    # Use the Lewis formula to compute the function to integrate
    f = 1 / (2 * np.pi) * phi(-xi - 1j / 2) / (xi ** 2 + 1 / 4)
    f_tilde = f * np.exp(-1j * z_1 * d_xi * np.arange(N))

    # Compute the FFT
    FFT = np.fft.fft(f_tilde)

    # Compute the prefactor
    prefactor = d_xi * np.exp(-1j * xi_1 * z)

    # Compute the integral by multiplying by the prefactor
    I = prefactor * FFT

    # Check that the imaginary part is close to zero
    if np.max(np.abs(np.imag(I))) > 1e-3:
        warning_msg = 'Imaginary part of the integral is not close to zero: ' \
                      f'{np.max(np.abs(np.imag(I)))} at iteration number: {M}'
        print(warning_msg)

    # Get only the real part
    I = np.real(I)

    # Interpolate the values
    interp_func = interp1d(z, I, kind='linear', fill_value='extrapolate')
    I_interpolated = interp_func(queryPoints)

    return I_interpolated



# Function to compute RMSE
def compute_rmse(Market, TTM, sigma, kappa, theta, M, dz, flag):
    """
    This function computes the root mean squared error (RMSE) between the model and the market prices for each
    maturity and each strike. The calibration is performed considering both markets.

    INPUTS
    Market: dictionary containing the market data
    TTM: array of expiries
    sigma: volatility
    kappa: volatility of the volatility
    theta: skewness of the volatility
    M: N = 2^M is the number of points in the grid
    dz: grid spacing

    OUTPUTS
    rmse_tot: total RMSE (root mean squared error) between the model and the market prices
    """

    # Initialize rmse array
    rmse_vett = np.zeros(len(TTM))

    # Cycle over expiries
    for ii in range(min(len(TTM), 19)):
        # Import data from the Market dictionary
        F0 = Market.F0[ii]
        strikes = Market.strikes[ii]
        B0 = Market.B_bar[ii]
        put = Market.midPut[ii]
        call = Market.midCall[ii]

        # Compute the log-moneyness
        log_moneyness = np.log(F0 / strikes)

        # Compute the call prices via Lewis formula
        callPrices = callIntegral(B0, F0, sigma, kappa, theta, TTM[ii], log_moneyness, M, dz, flag)

        # Compute the put prices via put-call parity
        putPrices = callPrices - B0 * (F0 - strikes)

        # Extract the model prices for calls and puts
        # Find indexes
        OTM_put_index = np.sum(strikes <= F0)
        OTM_call_index = OTM_put_index +1

        # Call prices for OTM options
        OTM_call_model = callPrices[OTM_call_index:]

        # Put prices for OTM options
        OTM_put_model = putPrices[:OTM_put_index]

        # Call prices for OTM options
        OTM_call_market = call[OTM_call_index:]

        # Put prices for OTM options
        OTM_put_market = put[:OTM_put_index]

        # Compute the RMSE
        rmse_vett[ii] = np.sqrt(np.mean(np.square(
            np.concatenate((OTM_put_model, OTM_call_model)) - np.concatenate((OTM_put_market, OTM_call_market)))))

    # Compute the total RMSE
    # rmse_tot = np.sum(weights * rmse_vett)
    rmse_tot = np.sum(rmse_vett)

    return rmse_tot



# Function to compute objective function
def objective_function(p, TTM_EU, TTM_US, w_EU, w_US, Market_EU, Market_US, M, dz, flag):
    """
    This function computes the objective function for the calibration of the
    model parameters. The objective function is the sum of the root mean
    squared errors (RMSE) between the model and the market prices for each
    maturity and each strike. The calibration is performed considering both
    the EUROStoxx50 and the S&P500 markets.

    INPUTS
    p: vector of model parameters
    TTM_EU: vector of maturities for the EUROStoxx50 options
    TTM_US: vector of maturities for the S&P500 options
    w_EU: weight for the EUROStoxx50 market
    w_US: weight for the S&P500 market
    Market_EU: structure containing the EUROStoxx50 market data
    Market_US: structure containing the S&P500 market data
    M: N = 2^M is the number of points in the grid
    dz: grid spacing

    OUTPUTS
    obj: objective function
    """

    # Call the parameters
    sigma_EU = p[0]
    kappa_EU = p[1]
    theta_EU = p[2]
    sigma_US = p[3]
    kappa_US = p[4]
    theta_US = p[5]

    # Compute the rmse for the EU Market
    rmseEU = compute_rmse(Market_EU, TTM_EU, sigma_EU, kappa_EU, theta_EU, M, dz, flag)

    # Compute the rmse for the US Market
    rmseUS = compute_rmse(Market_US, TTM_US, sigma_US, kappa_US, theta_US, M, dz, flag)

    # Compute the objective function
    obj = w_EU * rmseEU + w_US * rmseUS

    return obj


# Function to compute constraints


def constraint(x, alpha):
    """
    This is the constraint function for the "fmincon" Calibration.

    INPUTS:
    x: vector of parameters [sigma, kappa, theta, sigma_us, kappa_us, theta_us]
    alpha: exponent of the model

    OUTPUTS:
    c: inequality constraint
    ceq: equality constraint
    """

    # Inequality constraint
    c = []

    # Equality constraint
    ceq = x[0] ** 2 / (x[2] ** 2 * x[1]) - x[3] ** 2 / (x[5] ** 2 * x[4])

    return c, ceq