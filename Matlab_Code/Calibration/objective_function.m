function obj = objective_function(p, TTM_EU, TTM_US, w_EU, w_US, Market_EU, Market_US, M, dz, alpha, flag)
% This function computes the objective function for the calibration of the
% model parameters. The objective function is the sum of the root mean
% squared errors (RMSE) between the model and the market prices for each
% maturity and each strike. The calibation is performed considering both
% the EUROStoxx50 and the S&P500 markets.
%
% INPUTS
% p: vector of model parameters
% TTM_EU: vector of maturities for the EUROStoxx50 options
% TTM_US: vector of maturities for the S&P500 options
% w_EU: vweight for the EUROStoxx50 market
% w_US: weight for the S&P500 market
% Filtered_EU_Market: structure containing the EUROStoxx50 market data
% Filtered_US_Market: structure containing the S&P500 market data
% M: N = 2^M is the number of points in the grid
% dz: grid spacing
% alpha: Model selection parameter (NIG --> alpha = 0.5)
%                                  (VG  --> alpha ~ 0)
%
% OUTPUTS
% obj: objective function

% Call the parameters
sigma_EU = p(1);
kappa_EU = p(2);
theta_EU = p(3);
sigma_US = p(4);
kappa_US = p(5);
theta_US = p(6);

% Compute the rmse for the EU Market
rmseEU = compute_rmse(Market_EU, TTM_EU, sigma_EU, kappa_EU, theta_EU, alpha, M, dz, flag);

% Compute the rmse for the US Market
rmseUS = compute_rmse(Market_US, TTM_US, sigma_US, kappa_US, theta_US, alpha, M, dz, flag);

% Compute the objective function
obj = w_EU * rmseEU + w_US * rmseUS;

end