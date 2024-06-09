function obj = objective_function(p, TTM_EU, TTM_US, w_EU, w_US, Market_EU, Market_US, M, dz, flag, flag_rmse, flag_timeWindow)
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
% flag: model selection NIG or VG
% flag_rmse: RMSE or RMSE2
%
% OUTPUTS
% obj: objective function

%This function incorporates different methodologies for computing the RMSE (Root Mean Square Error). 
%               As a first step, it is possible to use two different functions:
%      RMSE      -> compute_rmse    : This function implements a simple calculation of the RMSE,
%                                       with or without weight adjustments.
%      RMSE2     -> compute_rmse_2  : This function calculates the RMSE by checking the bid-ask spread. 
%                                       Only prices that are outside this interval are used to compute the error.

% Call the parameters
sigma_EU = p(1);
kappa_EU = p(2);
theta_EU = p(3);
sigma_US = p(4);
kappa_US = p(5);
theta_US = p(6);

% To use compute_rmse_2 where only errors outside the bid-ask spred, change the function below from
%   compute_rmse ----------> compute_rmse_2
if strcmp(flag_rmse, 'RMSE')
    % Compute the rmse for the EU Market
    rmseEU = compute_rmse(Market_EU, TTM_EU, sigma_EU, kappa_EU, theta_EU, M, dz, flag, flag_timeWindow);

    % Compute the rmse for the US Market
    rmseUS = compute_rmse(Market_US, TTM_US, sigma_US, kappa_US, theta_US, M, dz, flag, flag_timeWindow);
elseif strcmp(flag_rmse, 'RMSE2')
    % Compute the rmse for the EU Market
    rmseEU = compute_rmse_2(Market_EU, TTM_EU, sigma_EU, kappa_EU, theta_EU, M, dz, flag);

    % Compute the rmse for the US Market
    rmseUS = compute_rmse_2(Market_US, TTM_US, sigma_US, kappa_US, theta_US, M, dz, flag);
else
    disp('Error: flag_rmse must be RMSE or RMSE2')
end
% Compute the objective function
obj = w_EU * rmseEU + w_US * rmseUS;

end