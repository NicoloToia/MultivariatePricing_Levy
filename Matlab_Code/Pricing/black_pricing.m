function price = black_pricing(Market_EU, S0_US, sigma_EU, sigma_US, setDate, targetDate, MeanBMs, rho, N_sim)
% This function computes the price of a derivative with the following payoff:
% Payoff = max(S1(t) - S1(0), 0)*I(S2(t) < 0.95*S2(0))
%
% INPUTS
% Market_EU: market data of the European asset
% Market_US: market data of the US asset
% sigma_EU: volatility of the European asset
% sigma_US: volatility of the US asset
% discounts: market discounts
% ttm: time to maturity
% MeanBMs: mean of the Brownian motions
% rho: correlation between the Brownian motions
% % N_sim: number of simulations
%
% OUTPUTS
% price: price of the derivative

% Recall variables from the market data
S0_EU = Market_EU.spot;
Expiries = datenum([Market_EU.datesExpiry]');
B_bar = [Market_EU.B_bar.value]';

% Compute the discount
discount = intExtDF(B_bar, Expiries, targetDate);

% Compute the forward prices
F0_EU = S0_EU/discount;
F0_US = S0_US/discount;

% Define the Covariance matrix
cov_matrix = [1 rho; rho 1];

% Generate the random numbers
Z = mvnrnd(MeanBMs, cov_matrix, N_sim);

% Compute the time to maturity
ACT_365 = 3;
ttm = yearfrac(setDate, targetDate, ACT_365);

% Simulate the assets via GBM (Geometric Brownian Motion)
S1_EU = F0_EU*exp((- 0.5*sigma_EU^2)*ttm + sigma_EU*sqrt(ttm)*Z(:,1));
S1_US = F0_US*exp((- 0.5*sigma_US^2)*ttm + sigma_US*sqrt(ttm)*Z(:,2));

% Compute the payoff
Indicator_function = (S1_US < 0.95*S0_US);
payoff = max(S1_EU - S0_EU, 0).*Indicator_function;

% Compute the price
price = discount*mean(payoff);

% Confidence interval
a = 0.01;
CI = norminv(1-a)*std(payoff)/sqrt(N_sim);
priceCI = [price - CI, price + CI];

% Display the results
fprintf('------------------------------------------------------------------\n');
fprintf('The price of the derivative via Black model is: %.4f\n', price);
fprintf('The confidence interval is: [%.4f, %.4f]\n', priceCI(1), priceCI(2));
fprintf('------------------------------------------------------------------\n');

end