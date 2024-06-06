# ---- THE BIG LÉVY PROJECT: Multivariate Pricing
# Final Project Financial Engineering 2024
# Professors: Roberto Baviera & Michele Azzone
# Group 2B
# Giacomo Manfredi  CP: 10776946
# Francesco Stillo  CP: 10698518
# Nicolò Toia       CP: 10628899
#


#%% ----  IMPORT PACKAGES ANF FUNCTION FILES
import numpy as np
from datetime import datetime
import scipy.io
import matplotlib.pyplot as plt
from generals import OptionMarketData, fwd_Bbar, yearfrac_act_365
from generals import compute_ImpVol, select_OTM
from generals import sens_Delta, Filter

#%% ---- FIX THE SEED
np.random.seed(42)  # the answer to everything

#%%  ----  IMPORT DATA
# Set the settlement date and vectors of dates
settlement = datetime.strptime("07/09/2023", "%m/%d/%Y")

dates_array0 = [
    '2023-07-21', '2023-08-18', '2023-09-15', '2023-10-20',
    '2023-11-17', '2023-12-15', '2024-01-19', '2024-02-16',
    '2024-03-15', '2024-06-21', '2024-09-20', '2024-12-20', '2025-06-20'
]
dates_EU = [datetime.strptime(date, '%Y-%m-%d') for date in dates_array0]

dates_array1 = [
    '2023-07-21', '2023-08-18', '2023-09-15', '2023-10-20',
    '2023-11-17', '2023-12-15', '2024-01-19', '2024-02-16',
    '2024-03-15', '2024-04-19', '2024-05-17', '2024-06-21',
    '2024-07-19', '2024-09-20', '2024-12-20', '2025-06-20',
    '2025-12-19', '2026-12-18', '2027-12-17', '2028-12-15'
]
dates_US = [datetime.strptime(date, '%Y-%m-%d') for date in dates_array1]


# ---- LOAD THE DATASET
markets = scipy.io.loadmat('OptionData.mat')
Mkt_EU = markets['mkt_EU']
Mkt_US = markets['mkt']
del markets
returns = scipy.io.loadmat('SPXSX5Ereturns.mat')


#%% ---- CREATE MARKET_EU & MARKET_US
# Function to extract data from the nested array structure
def extract_data_EU(field):
    return [arr[0][0].tolist() for arr in Mkt_EU[field][0, 0][0]]

def extract_data_US(field):
    return [arr[0][0].tolist() for arr in Mkt_US[field][0, 0][0]]

# Extracting data for each field
callBid_EU = extract_data_EU('callBid')
callAsk_EU = extract_data_EU('callAsk')
putAsk_EU = extract_data_EU('putAsk')
putBid_EU = extract_data_EU('putBid')
strikes_EU = extract_data_EU('strikes')
spot_EU = Mkt_EU['spot'][0][0][0][0]

# Create the OptionMarketData object
Market_EU = OptionMarketData(
    datesExpiry=dates_EU,
    callBid=callBid_EU,
    callAsk=callAsk_EU,
    putAsk=putAsk_EU,
    putBid=putBid_EU,
    strikes=strikes_EU,
    spot=spot_EU,
)

# Extracting data for each field
callBid_US = extract_data_US('callBid')
callAsk_US = extract_data_US('callAsk')
putAsk_US = extract_data_US('putAsk')
putBid_US = extract_data_US('putBid')
strikes_US = extract_data_US('strikes')
spot_US = Mkt_US['spot'][0][0][0][0]

# Create the OptionMarketData object
Market_US = OptionMarketData(
    datesExpiry=dates_US,
    callBid=callBid_US,
    callAsk=callAsk_US,
    putAsk=putAsk_US,
    putBid=putBid_US,
    strikes=strikes_US,
    spot=spot_US,
)


#%% ---- COMPUTE DISCOUNT FACTORS AND FORWARD PRICES FROM OPTION DATA
# Compute the market discount factors and forward prices
Market_EU = fwd_Bbar(Market_EU)
Market_US = fwd_Bbar(Market_US)

#%% ---- CREATE AUXILIARY VARIABLES AND COMPUTE MARKET ZERO RATES


# Extract Markets discouts
discounts_EU = Market_EU.B_bar
discounts_US = Market_US.B_bar

# Extract forward prices
F0_EU = Market_EU.F0
F0_US = Market_US.F0

# Plot fwd prices
# plt.figure()
# plt.plot(Market_EU.datesExpiry, F0_EU, 'b', linewidth=1)
# plt.plot(Market_US.datesExpiry, F0_US, 'r', linewidth=1)
# plt.ylabel('Forward Prices')
# plt.title('Forward Prices for the EURO STOXX 50')
# plt.legend(['Forward Prices EU', 'Forward Prices EU'], loc='best')
# plt.grid(True)
# plt.show()

# Compute Time to Maturity (TTM) in yearfrac
TTM_EU = yearfrac_act_365(settlement, Market_EU.datesExpiry)
TTM_US = yearfrac_act_365(settlement, Market_US.datesExpiry)

# Compute market rates
rates_EU = -np.log(discounts_EU) / TTM_EU
rates_US = -np.log(discounts_US) / TTM_US


# %% ---- COMPUTE IMPLIED VOLATILITIES & SELECT OUT OF THE MONEY (OTM) OPTIONS
# Compute the implied volatilities for the EU market
Market_EU = compute_ImpVol(Market_EU, TTM_EU, rates_EU)
# Compute the implied volatilities for the US market
#Market_US = compute_ImpVol(Market_US, TTM_US, rates_US)

# Select the OTM implied volatilities for the EU market
Market_EU = select_OTM(Market_EU)

# Select the OTM implied volatilities for the US market
# Market_US = select_OTM(Market_US)

#%% ---- FILTERING

# Compute the delta sensitivity for the EU market
Market_EU = sens_Delta(Market_EU, TTM_EU, rates_EU)
# Compute the delta sensitivity for the US market
# Market_US = sens_Delta(Market_US, TTM_US, rates_US)

# Create a new struct for the EU market with the filtered options
Market_EU_filtered = Filter(Market_EU)

# Create a new struct for the US market with the filtered options
# Market_US_filtered = Filter(Market_US)
