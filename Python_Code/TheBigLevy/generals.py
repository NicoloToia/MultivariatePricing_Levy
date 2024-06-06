# import packages
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Define classes for markets EU and US
class OptionMarketData:
    def __init__(self, datesExpiry, callBid, callAsk, putAsk, putBid, strikes, spot):
        self.datesExpiry = datesExpiry
        self.callBid = callBid
        self.callAsk = callAsk
        self.putAsk = putAsk
        self.putBid = putBid
        self.strikes = strikes
        self.spot = spot
        # Initialize new attributes with None or appropriate empty structures
        self.B_bar = [None] * len(datesExpiry)
        self.F = [None] * len(datesExpiry)
        self.FBid = [None] * len(datesExpiry)
        self.FAsk = [None] * len(datesExpiry)
        self.F0 = [None] * len(datesExpiry)
        self.midCall = [None] * len(datesExpiry)
        self.midPut = [None] * len(datesExpiry)
        self.ImpVol_call = [None] * len(datesExpiry)
        self.ImpVol_put = [None] * len(datesExpiry)
        self.OTM_ImpVol_put = [None] * len(datesExpiry)
        self.OTM_ImpVol_call = [None] * len(datesExpiry)
        self.OTM_ImpVol = [None] * len(datesExpiry)
        self.sensDelta_call = [None] * len(datesExpiry)
        self.sensDelta_put = [None] * len(datesExpiry)


    def __repr__(self):
        return (f"OptionMarketData(datesExpiry={len(self.datesExpiry)}, callBid={len(self.callBid)}, "
                f"callAsk={len(self.callAsk)}, putAsk={len(self.putAsk)}, putBid={len(self.putBid)}, "
                f"strikes={len(self.strikes)}, spot={len(self.spot)}, B_bar={len(self.B_bar)}, F={len(self.F)},"
                f"FBid={len(self.FBid)}, FAsk={len(self.FAsk)}, F0={len(self.F0)}, "
                f"midCall={len(self.midCall)}, midPut={len(self.midPut)}, "
                f"ImpVol_call ={len(self.ImpVol_call)}, ImpVol_put={len(self.ImpVol_put)},"
                f"OTM_ImpVol_put ={len(self.OTM_ImpVol_put)}, OTM_ImpVol_call={len(self.OTM_ImpVol_call)},"
                f"OTM_ImpVol ={len(self.OTM_ImpVol)}, sensDelta_call={len(self.sensDelta_call)},"
                f"sensDelta_put ={len(self.sensDelta_put)}"
                )


class Filtered_OptionMarketData:
    def __init__(self, datesExpiry, callBid, callAsk, putAsk, putBid, strikes, spot, B_bar, F0, midCall, midPut, OTM_ImpVol):
        self.datesExpiry = datesExpiry
        self.callBid = callBid
        self.callAsk = callAsk
        self.putAsk = putAsk
        self.putBid = putBid
        self.strikes = strikes
        self.spot = spot
        self.B_bar = B_bar
        self.F0 = F0
        self.midCall = midCall
        self.midPut = midPut
        self.OTM_ImpVol = OTM_ImpVol


    def __repr__(self):
        return (f"OptionMarketData(datesExpiry={len(self.datesExpiry)}, callBid={len(self.callBid)}, "
                f"callAsk={len(self.callAsk)}, putAsk={len(self.putAsk)}, putBid={len(self.putBid)}, "
                f"strikes={len(self.strikes)}, spot={len(self.spot)}, B_bar={len(self.B_bar)}, F0={len(self.F0)}, "
                f"midCall={len(self.midCall)}, midPut={len(self.midPut)}, OTM_ImpVol ={len(self.OTM_ImpVol)}"
                )


# Functions to compute Market discount factors & Forward prices
def fwd_Bbar(Market):
    # Loop over the maturities
    for ii in range(len(Market.datesExpiry)):
        # Compute the synthetic forward: G_bid(K), G_ask(K), G(K)
        GBid = np.array(Market.callBid[ii]) - np.array(Market.putAsk[ii])
        GAsk = np.array(Market.callAsk[ii]) - np.array(Market.putBid[ii])
        G = (GBid + GAsk) / 2
        # Compute G_hat and K_hat (sample mean of G(K) and K's)
        G_hat = np.mean(G)
        K_hat = np.mean(Market.strikes[ii])


        # Compute market discount factors between t0 and T_i
        Num = np.sum((np.array(Market.strikes[ii]) - K_hat) * (G - G_hat))
        Den = np.sum((np.array(Market.strikes[ii]) - K_hat) ** 2)
        Market.B_bar[ii] = - Num / Den
        # Compute forward price F, F_bid and F_ask
        Market.F[ii] = G / Market.B_bar[ii] + np.array(Market.strikes[ii])
        Market.FBid[ii] = GBid / Market.B_bar[ii] + np.array(Market.strikes[ii])
        Market.FAsk[ii] = GAsk / Market.B_bar[ii] + np.array(Market.strikes[ii])

        # Compute the average of forward prices for each maturity: F0
        Market.F0[ii] = np.mean(Market.F[ii])

        # Compute the mid prices of the options
        Market.midCall[ii] = (np.array(Market.callBid[ii]) + np.array(Market.callAsk[ii])) / 2
        Market.midPut[ii] = (np.array(Market.putBid[ii]) + np.array(Market.putAsk[ii])) / 2

    return Market


# Compute yearfrac
def yearfrac_act_365(start_date, end_date):
    """
    This function calculates the year fraction in convenction ACT/365

    INPUT
    start_date: starting date
    end_date: ending date

    OUTPUT
    return year fraction in convenction ACT/365
    """
    delta_year = [x - start_date for x in end_date]
    delta_days = [x.days for x in delta_year]
    return np.array([x/365 for x in delta_days])


# Function to compute Black price of a call
def black_call(F, K, T, r, sigma):
    """
    This function computes the Black price of a call option
    :param F: forward price
    :param K: strike
    :param T: ttm
    :param r: rates
    :param sigma: volatility
    :return: price of the option
    """
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return call_price

# FUnction to compute Black price of a put
def black_put(F, K, T, r, sigma):
    """
    This function computes the Black price of a put option
    :param F: forward price
    :param K: strike
    :param T: ttm
    :param r: rates
    :param sigma: volatility
    :return: price of the option
    """
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return put_price


# Functions to compute implied vol
def implied_volatility_call(F, K, T, r, market_price):
    def objective_function(sigma):
        return black_call(F, K, T, r, sigma) - market_price

    return brentq(objective_function, -1e-6, 5)


def implied_volatility_put(F, K, T, r, market_price):
    def objective_function(sigma):
        return black_put(F, K, T, r, sigma) - market_price
    return brentq(objective_function, -1e-6, 5)

def compute_ImpVol(Market, TTM, rates):
    """
    This function computes the implied volatilities for real - world Market & Market model data

    :param Market: structure with the market data
    :param TTM: time to maturity in year fractions
    :param rates: market zero rates
    :return: structure with the market data and the implied volatilities

    """

    for ii in range(len(Market.datesExpiry)):
        F0_value = Market.F0[ii]
        strikes = Market.strikes[ii]
        mid_call_prices = Market.midCall[ii]
        mid_put_prices = Market.midPut[ii]

        Market.ImpVol_call[ii] = [None] * len(strikes)
        Market.ImpVol_put[ii] = [None] * len(strikes)

        prova1 = []
        prova2 = []

        for jj in range(len(strikes)):
            vol_call = implied_volatility_call(F0_value, strikes[jj], TTM[ii], rates[ii], mid_call_prices[jj])
            vol_put = implied_volatility_put(F0_value, strikes[jj], TTM[ii], rates[ii], mid_put_prices[jj])
            prova1.append(vol_call)
            prova2.append(vol_put)

        Market.ImpVol_call[ii] = prova1
        Market.ImpVol_put[ii] = prova2
    return Market


# Function to select OTM options
def select_OTM(Market):
    """

    This function selects the out of the money options from the market data
    and builds a new implied volatility smile

    INPUTS
    Market: structure with the market data

    OUTPUTS
    Market: structure with the market data and the OTM implied volatilities

    """

    # Call the needed variables from the struct
    F0 = Market.F0

    # Cycle over the different expiries and for each of them build the OTM
    # The smile is constructed considering only the OTM options:
    #   1. If F0 >= K, use put options
    #   2. If F0 < K, use call options

    for ii in range(len(Market.datesExpiry)):

        # Find the index of the strike before the forward price
        strikes = Market.strikes[ii]
        idx = next((i for i, x in enumerate(strikes) if x >= F0[ii]), len(strikes))
        # Create the smile for the i-th maturity
        Market.OTM_ImpVol_put[ii] = Market.ImpVol_put[ii][:idx]
        Market.OTM_ImpVol_call[ii] = Market.ImpVol_call[ii][idx:]
        Market.OTM_ImpVol[ii] = Market.OTM_ImpVol_put[ii] + Market.OTM_ImpVol_call[ii]

        # Check for NaN values in the implied volatilities
        if any(np.isnan(Market.OTM_ImpVol[ii])):
            raise ValueError('NaN values in the implied volatilities')

    return Market


# Function to compute delta-sensitivities
def blsdelta(S, K, r, T, sigma, q):
    """
    Compute the Black-Scholes delta of an option.

    Parameters:
    S : spot price of the underlying asset
    K : strike price
    r : risk-free interest rate
    T : time to maturity
    sigma : volatility
    q : dividend yield

    Returns:
    delta_call : delta of the call option
    delta_put :  delta of the put option
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta_call = np.exp(-q * T) * norm.cdf(d1)
    delta_put = np.exp(-q * T) * (norm.cdf(d1) - 1)
    return delta_call, delta_put

def sens_Delta(Market, TTM, rates):
    """
    This function computes the delta sensitivity for the market data.

    INPUTS:
    Market: structure with the market data
    TTM: time to maturity in year fractions
    rates: market zero rates

    OUTPUT:
    Market: structure with the market data and the delta sensitivity
    """

    # Call the needed variables from the struct
    S0 = Market.spot
    # Null dividend
    d = 0

    # Cycle over the different expiries and for each of them compute the delta sensitivities
    for ii in range(len(TTM)):

        strikes = Market.strikes[ii]
        OTM_ImpVol = Market.OTM_ImpVol[ii]

        Market.sensDelta_call[ii] = [None] * len(strikes)
        Market.sensDelta_put[ii] = [None] * len(strikes)

        call_deltas = []
        put_deltas = []

        for jj in range(len(strikes)):
            call_delta, put_delta = blsdelta(S0, strikes[jj], rates[ii], TTM[ii], OTM_ImpVol[jj], d)
            call_deltas.append(call_delta)
            put_deltas.append(put_delta)

        Market.sensDelta_call[ii] = call_deltas
        Market.sensDelta_put[ii] = put_deltas

    return Market



# Function to filter the market
def Filter(Market):
    """
    This function filters the market data based on the delta sensitivities.
    If the delta is between 0.1 and 0.9 for calls and -0.9 and -0.1 for puts, the option is kept.

    INPUTS
    market: dictionary containing the market data

    OUTPUTS
    Filtered_Market: dictionary containing the filtered market data
    """

    # datesExpiry, callBid, callAsk, putAsk, putBid, strikes, spot, Volume_call, Volume_put, B_bar, F0, midCall, midPut, OTM_ImpVol

    # Past what is unchanged
    datesExpiry = Market.datesExpiry
    spot = Market.spot
    B_bar = Market.B_bar
    F0 = Market.F0

    # Initialize empty lists
    callBid = []
    callAsk = []
    putAsk = []
    putBid = []
    strikes = []
    midCall = []
    midPut = []
    OTM_ImpVol = []

    # Cycle through the expiries
    for ii in range(len(Market.datesExpiry)):

        # Get the call and put deltas
        call_delta = np.array(Market.sensDelta_call[ii])
        put_delta = np.array(Market.sensDelta_put[ii])

        # Filter deltas for the options with delta between 0.1 and 0.9 for calls and -0.9 and -0.1 for puts
        valid_call = (call_delta >= 0.1) & (call_delta <= 0.9)
        valid_put = (put_delta >= -0.9) & (put_delta <= -0.1)

        # call Bid and Ask
        callBid.append([element for element, condition in zip(Market.callBid[ii], valid_call) if condition])
        callAsk.append([element for element, condition in zip(Market.callAsk[ii], valid_call) if condition])

        # Put Bid and Ask
        putBid.append([element for element, condition in zip(Market.putBid[ii], valid_put) if condition])
        putAsk.append([element for element, condition in zip(Market.putAsk[ii], valid_put) if condition])

        # Strikes
        valid_strikes = valid_call & valid_put
        strikes.append([element for element, condition in zip(Market.strikes[ii], valid_strikes) if condition])

        # mid prices
        midCall.append([element for element, condition in zip(Market.midCall[ii], valid_call) if condition])
        midPut.append([element for element, condition in zip(Market.midPut[ii], valid_put) if condition])

        # Implied volatilities
        OTM_ImpVol.append([element for element, condition in zip(Market.OTM_ImpVol[ii], valid_call) if condition])

    Filtered_Market = Filtered_OptionMarketData(
        datesExpiry = datesExpiry,
        callBid = callBid,
        callAsk = callAsk,
        putAsk = putAsk,
        putBid = putBid,
        strikes = strikes,
        spot = spot,
        B_bar = B_bar,
        F0 = F0,
        midCall = midCall,
        midPut = midPut,
        OTM_ImpVol = OTM_ImpVol
    )

    return Filtered_Market