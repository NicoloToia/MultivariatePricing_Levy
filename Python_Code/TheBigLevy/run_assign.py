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
import pandas as pd
from pandas.tseries.offsets import BusinessDay
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

from generals import OptionMarketData, fwd_Bbar, yearfrac_act_365
from generals import compute_ImpVol, select_OTM
from generals import sens_Delta, Filter
from calibration import objective_function, constraint
from calibration import Calibrated_OptionMarketData
from calibration_black import Black_OptionMarketData, black_obj
from pricing import black_pricing, closedFormula, levy_pricing2

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
plt.figure()
plt.plot(Market_EU.datesExpiry, F0_EU, 'b', linewidth=1)
plt.plot(Market_US.datesExpiry, F0_US, 'r', linewidth=1)
plt.ylabel('Forward Prices')
plt.title('Forward Prices for the EURO STOXX 50')
plt.legend(['Forward Prices EU', 'Forward Prices EU'], loc='best')
plt.grid(True)
plt.show()

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
Market_US = compute_ImpVol(Market_US, TTM_US, rates_US)

# Select the OTM implied volatilities for the EU market
Market_EU = select_OTM(Market_EU)

# Select the OTM implied volatilities for the US market
Market_US = select_OTM(Market_US)


#%% ---- FILTERING

# Compute the delta sensitivity for the EU market
Market_EU = sens_Delta(Market_EU, TTM_EU, rates_EU)
# Compute the delta sensitivity for the US market
Market_US = sens_Delta(Market_US, TTM_US, rates_US)

# Create a new struct for the EU market with the filtered options
Market_EU_filtered = Filter(Market_EU)

# Create a new struct for the US market with the filtered options
Market_US_filtered = Filter(Market_US)


#%% ---- CALIBRATION

# Define the weight of both markets (EU and US)
w_EU = spot_EU / (spot_EU + spot_US)
w_US = spot_US / (spot_EU + spot_US)

# Set the Fast Fourier Transform (FFT) parameters
M_fft = 15
dz_fft = 0.0025

# Calibrate the NIG parameters for the two markets (EU and US)
# sigma_EU = p[0]
# kappa_EU = p[1]
# theta_EU = p[2]
# sigma_US = p[3]
# kappa_US = p[4]
# theta_US = p[5]

# Fix the flag to use NIG model
flag = 'NIG'

# Define the objective function
def obj_fun(p):
    return objective_function(p, TTM_EU, TTM_US, w_EU, w_US, Market_EU_filtered, Market_US_filtered, M_fft, dz_fft, flag)

# Linear costraints
A = np.array([
    [-1, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, -1, 0],
])
lb_ineq = -np.inf * np.ones(4)
ub_ineq = np.zeros(4)  # Right-hand side of the inequality

linear_constraint = LinearConstraint(A, lb_ineq, ub_ineq)


# Initial guess
p0 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Nonlinear constraints
def constraint_ineq(x):
    c, _ = constraint(x, 0.5)
    return c

def constraint_eq(x):
    _, ceq = constraint(x, 0.5)
    return ceq

nonlinear_constraints = [
    {'type': 'ineq', 'fun': constraint_ineq},
    {'type': 'eq', 'fun': constraint_eq}
]


# Define the bounds
bounds = [(0, None),  # x1 >= 0
          (0, None),  # x2 >= 0
          (None, None),  # x3 non limitato
          (0, None),  # x4 >= 0
          (0, None),  # x5 >= 0
          (None, None)]  # x6 non limitato

# Options
options = {
    'disp': True,
}


# Optimization
result = minimize(obj_fun, p0, bounds=bounds, constraints=[linear_constraint] + nonlinear_constraints, options=options)
calibrated_param = result.x



# Print the results
print('---------------------------------------------------------------------')
print('The optimal parameters are:')
print(f'sigma_EU = {calibrated_param[0]}')
print(f'kappa_EU = {calibrated_param[1]}')
print(f'theta_EU = {calibrated_param[2]}')
print(f'sigma_US = {calibrated_param[3]}')
print(f'kappa_US = {calibrated_param[4]}')
print(f'theta_US = {calibrated_param[5]}')
print('---------------------------------------------------------------------')

# Rename the calibrated parameters for the EU market
sigma_EU = calibrated_param[0]
kappa_EU = calibrated_param[1]
theta_EU = calibrated_param[2]

# Rename the calibrated parameters for the US market
sigma_US = calibrated_param[3]
kappa_US = calibrated_param[4]
theta_US = calibrated_param[5]


#%% ---- NEW STRUCT FOR MARKET MODEL

# For EU Market
Market_EU_calibrated = Calibrated_OptionMarketData(
    datesExpiry=Market_EU_filtered.datesExpiry,
    strikes=Market_EU_filtered.strikes,
    spot=Market_EU_filtered.spot,
    B_bar=Market_EU_filtered.B_bar,
    F0=Market_EU_filtered.F0,
    sigma=sigma_EU,
    kappa=kappa_EU,
    theta=theta_EU
)

# For US Market
Market_US_calibrated = Calibrated_OptionMarketData(
    datesExpiry=Market_US_filtered.datesExpiry,
    strikes=Market_US_filtered.strikes,
    spot=Market_US_filtered.spot,
    B_bar=Market_US_filtered.B_bar,
    F0=Market_US_filtered.F0,
    sigma=sigma_US,
    kappa=kappa_US,
    theta=theta_US
)

#%% ---- COMPUTE nu_Z USING CLOSE FORMULAS

# Import rho from MATLAB script
rho = 0.24264
nu_Z = np.sqrt(kappa_EU*kappa_US)/rho
nu_EU = (kappa_EU * nu_Z) / (nu_Z - kappa_EU)
nu_US = (kappa_US * nu_Z) / (nu_Z - kappa_US)

# Stampa i risultati
print('---------------------------------------------------------------------')
print(f'nu_EU = {nu_EU}')
print(f'nu_US = {nu_US}')
print(f'nu_Z = {nu_Z}')
print('---------------------------------------------------------------------')

#%% ---- COMPUTE HISTORICAL CORRELATION

returns_annually = returns['Returns'][0, 0]['Annually']

# Compute historical correlation
correlation_matrix = np.corrcoef(returns_annually.T)
hist_corr = correlation_matrix[0, 1]

#%% ---- NEW STRUCT FOR MARKET MODEL (BLACK)
# EU Market
Market_EU_black = Black_OptionMarketData(
    datesExpiry=Market_EU_filtered.datesExpiry,
    strikes=Market_EU_filtered.strikes,
    spot=Market_EU_filtered.spot,
    B_bar=Market_EU_filtered.B_bar,
    F0=Market_EU_filtered.F0
)

# US Market
Market_US_black = Black_OptionMarketData(
    datesExpiry=Market_US_filtered.datesExpiry,
    strikes=Market_US_filtered.strikes,
    spot=Market_US_filtered.spot,
    B_bar=Market_US_filtered.B_bar,
    F0=Market_US_filtered.F0
)

#%% ---- BLACK CALIBRATION

# Import the results from matlab scripts
sigmaB_EU = 0.15688
sigmaB_US = 0.16405

# Print the results
print('---------------------------------------------------------------------')
print('The calibrated parameters are:')
print(f'sigmaB_EU = {sigmaB_EU}')
print(f'sigmaB_US = {sigmaB_US}')
print('---------------------------------------------------------------------')

# Add volatilities to the struct
Market_EU_black.sigma = sigmaB_EU
Market_US_black.sigma = sigmaB_US

#%% ---- PRICING: BLACK MODEL

target_date = settlement + pd.DateOffset(years=1)

# Function to check if a date is a business day
def is_business_day(date):
    return np.is_busday(date.strftime('%Y-%m-%d'))

# Adjust target date to next business day if it's not a business day
if not is_business_day(target_date):
    target_date = target_date + BusinessDay()

# Intialize the mean of the Brownian motions
MeanBMs = np.array([0, 0])

# Number of simulations
N_sim = int(1e7)

# Compute the price of the derivative using the Black model
price_black, CI_black = black_pricing(Market_US_black, Market_EU_black, settlement, target_date, MeanBMs, hist_corr, N_sim)

# Print the results
print("---------------------------------------------------------------------")
print(f"Price using Black model: {round(price_black,2)}")
print(f"Confidence interval: {[round(p,2) for p in CI_black]}")
print("---------------------------------------------------------------------")

#%% ---- PRICING USING SEMI-CLOSED FORMULA

price_closed_formula = closedFormula(Market_US_black, Market_EU_black, settlement, target_date, hist_corr)
print("---------------------------------------------------------------------")
print(f"Price using semi-closed Black formula: {round(price_closed_formula,2)}")
print("---------------------------------------------------------------------")

#%% ---- PRICING VIA THE LÉVY MODEL

[price_levy2, CI_levy2] = levy_pricing2(Market_US_calibrated, Market_EU_calibrated, settlement, target_date,
                                    kappa_US, kappa_EU, sigma_US, sigma_EU, theta_US, theta_EU, rho, N_sim, flag)

# Print the results
print("---------------------------------------------------------------------")
print(f"Price using Lévy model: {round(price_levy2,2)}")
print(f"Confidence interval: {[round(p,2) for p in CI_levy2]}")
print("---------------------------------------------------------------------")

