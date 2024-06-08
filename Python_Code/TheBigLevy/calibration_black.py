# import packages
import numpy as np
from generals import black_call, black_put


class Black_OptionMarketData:
    def __init__(self, datesExpiry, strikes, spot, B_bar, F0):
        self.datesExpiry = datesExpiry
        self.strikes = strikes
        self.spot = spot
        self.B_bar = B_bar
        self.F0 = F0
        # Initialize new attributes with None or appropriate empty structures
        self.sigma = [None]


    def __repr__(self):
        return (f"OptionMarketData(datesExpiry={len(self.datesExpiry)}, strikes={len(self.strikes)},"
                f"spot={len(self.spot)}, B_bar={len(self.B_bar)}, F0={len(self.F0)}"
                #f"sigma={len(self.sigma)}, kappa={len(self.kappa)}, theta={len(self.theta)}"
                )


# FUnction to compute the Black objective function
def black_obj(Market, maturity, sigma):
    """
    This function defines the objective function for the calibration of the 2-dimensional process using the Black model.

    Parameters:
    Market : dict
        Dictionary containing the market data.
    maturity : array_like
        Vector of maturities.
    sigma : float
        Volatility.

    Returns:
    float
        Objective function value.
    """
    # Initialize the RMSE (root mean square error) vector
    rmse_vett = np.zeros(len(maturity))

    # Cycle over the maturities
    for ii in range(len(Market.datesExpiry)):
        # EU market data from the structure
        F0 = Market.F0[ii]
        strikes = Market.strikes[ii]
        B0 = Market.B_bar[ii]
        put = Market.midPut[ii]
        call = Market.midCall[ii]

        # Compute the market zero rate
        rate = -np.log(B0) / maturity[ii]
        # Compute call and put prices via Black model
        callPrices= black_call(F0, strikes, rate, maturity[ii], sigma)
        putPrices = black_put(F0, strikes, rate, maturity[ii], sigma)

        # Extract the model prices for calls and puts
        # Find indexes
        OTM_put_index = np.sum(strikes <= F0)
        OTM_call_index = OTM_put_index +1

        # Call prices for OTM options
        OTM_call_model = callPrices[OTM_call_index:]

        # Put prices for OTM options
        OTM_put_model = putPrices[:OTM_put_index]

        # Extract the market prices for calls and puts
        # Call prices for OTM options
        OTM_call_market = call[OTM_call_index:]

        # Put prices for OTM options
        OTM_put_market = put[:OTM_put_index]

        # Compute the RMSE
        rmse_vett[ii] = np.sqrt(np.mean(np.square(
            np.concatenate((OTM_put_model, OTM_call_model)) - np.concatenate((OTM_put_market, OTM_call_market)))))

    # Objective function
    return np.sum(rmse_vett)