function obj = black_obj(sigma, maturity, Filtered_Market)
% OBJECTIVE_FUNCTION computes the objective function for the calibration of the 2-dimensional process
%
% INPUTS:
% p: parameters of the model
% maturity_EU: maturities of the EU options
% maturity_US: maturities of the US options
% w_EU: weight of the EU market
% w_US: weight of the US market
% Filtered_EU_Market: structure containing the EU market data
% Filtered_US_Market: structure containing the US market data
% M: number of points to use in the FFT
% dz: step size for the FFT
%
% OUTPUTS:
% obj: objective function value

% initialize of th errors
rmse = zeros(length(maturity),1);

N = 0;

for ii = 1:length(maturity)

    % EU market data from the structure
    F0 = Filtered_Market.F0(ii).value; % forward price
    strikes = Filtered_Market.strikes(ii).value; % strikes
    B0 = Filtered_Market.B_bar(ii).value; % discount factor
    Put = (Filtered_Market.putAsk(ii).prices + Filtered_Market.putBid(ii).prices)/2; % mid prices put
    Call = (Filtered_Market.callAsk(ii).prices + Filtered_Market.callBid(ii).prices)/2; % mid prices call

    % mkt zero rates
    rate = -log(B0)/maturity(ii);

    % compute call prices via black
    callPrices = blkprice(F0, strikes, rate, maturity(ii), sigma);
    
    % put-call paritybl
    putPrices = callPrices - B0*(F0 - strikes);
    
    % Objective function
    rmse(ii) = sqrt(sum(((putPrices - Put).*(strikes < F0)).^2) + sum(((callPrices - Call).*(strikes > F0)).^2));

    N = N + length(strikes);

end

rmse = sum(rmse)/N;

obj = rmse;

end