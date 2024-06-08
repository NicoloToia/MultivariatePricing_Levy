function [price, priceCI] = black_pricing(Market_US, Market_EU, setDate, targetDate, MeanBMs, rho, N_sim)
% This function computes the price of a derivative with the following payoff:
% Payoff = max(S1(t) - S1(0), 0)*I(S2(t) < 0.95*S2(0))
%
% INPUTS
% Market_US: market data of the US asset
% S0_US: spot price of the US asset
% sigma_US: volatility of the US asset
% sigma_EU: volatility of the European asset
% setDate: date of the simulation
% targetDate: target date of the simulation
% MeanBMs: mean of the Brownian motions
% rho: correlation between the Brownian motions
% N_sim: number of simulations
%
% OUTPUTS
% price: price of the derivative

% Recall variables from the market data

% Import the spot prices
S0_US = Market_US.spot;
S0_EU = Market_EU.spot;

% Import the expiries
Expiries_US = datenum([Market_US.datesExpiry]');
Expiries_EU = datenum([Market_EU.datesExpiry]');

% Import the market discounts B_bar
B_bar_US = [Market_US.B_bar.value]';
B_bar_EU = [Market_EU.B_bar.value]';

% Import the calibrated volatilities
sigma_US = Market_US.sigma;
sigma_EU = Market_EU.sigma;

% Compute the discount via interpolation
discount_US = intExtDF(B_bar_US, Expiries_US, targetDate);
discount_EU = intExtDF(B_bar_EU, Expiries_EU, targetDate);

% Compute the forward prices
F0_US = S0_US/discount_US;
F0_EU = S0_EU/discount_EU;

% Define the Covariance matrix
cov_matrix = [1 rho; rho 1];

% Generate correlated random variables from the multivariate normal distribution
Z = mvnrnd(MeanBMs, cov_matrix, N_sim);

% Compute the time to maturity
ACT_365 = 3;
ttm = yearfrac(setDate, targetDate, ACT_365);

% Simulate the assets via GBM (Geometric Brownian Motion)
S1_US = F0_US*exp((- 0.5*sigma_US^2)*ttm + sigma_US*sqrt(ttm)*Z(:,1));
S1_EU = F0_EU*exp((- 0.5*sigma_EU^2)*ttm + sigma_EU*sqrt(ttm)*Z(:,2));

% Coompute the payoff
Indicator_function = (S1_EU < 0.95*S0_EU);
payoff = max(S1_US - S0_US, 0).*Indicator_function;

% Compute the price
price = discount_US * mean(payoff);

% Confidence interval
a = 0.01;
CI = norminv(1-a)*std(payoff)/sqrt(N_sim);
priceCI = [price - CI, price + CI];

end
