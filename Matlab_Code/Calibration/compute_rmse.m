function rmse_tot = compute_rmse(Market, TTM, sigma, kappa, theta, alpha, M, dz)
% This function computes the root mean squared error (RMSE) between the model and the market prices for each
% maturity and each strike. The calibation is performed considering both markets.
%
% INPUTS 
% Market: struct containing the market data
% TTM: vector of expiries
% sigma: volatility
% kappa: volatility of the volatility
% theta: skewness of the volatility
% alpha: Model parameter (NIG --> alpha = 0.5)
% M: N = 2^M is the number of points in the grid
% dz: grid spacing
%
% OUTPUTS
%
% rmse_tot: total RMSE (root mean squared error) between the model and the market prices

% Initialize rmse vector
rmse_vett = zeros(length(TTM), 1);

% Compute weights to overweight short maturities errors on prices
weights = flip((TTM./TTM(end))/sum(TTM./TTM(end)));

% Cycle over expiries
for ii = 1:length(TTM)

    % Import data from the Market struct
    F0 = Market.F0(ii).value;
    strikes = [Market.strikes(ii).value]';
    B0 = Market.B_bar(ii).value;
    put = Market.midPut(ii).value;
    call = Market.midCall(ii).value;

    % Compute the log-moneyness
    log_moneyness = log(F0./strikes);

    % Compute the call prices via Lewis formula
    callPrices = callIntegral(B0, F0, alpha, sigma, kappa, theta, TTM(ii), log_moneyness, M, dz, 'FFT');

    % Compute the put prices via put-call parity
    putPrices = callPrices - B0*(F0 - strikes);

    % Extract the model prices for calls and puts
    % Find indexes
    OTM_put_index = sum((strikes <= F0) == 1);
    OTM_call_index = OTM_put_index + 1;
    
    % Call prices for OTM options
    OTM_call_model = callPrices(OTM_call_index:end);

    % Put prices for OTM options
    OTM_put_model = putPrices(1:OTM_put_index);

    % Call prices for OTM options
    OTM_call_market = call(OTM_call_index:end);

    % Put prices for OTM options
    OTM_put_market = put(1:OTM_put_index);

    % ******************************************************************
    % w_call = Market.Volume_call(ii).volume; 
    % w_put = Market.Volume_put(ii).volume;

    % % Pesi
    % pes_put_long = w_put.*(strikes <= F0);
    % pes_put = pes_put_long(pes_put_long~=0);
    % pes_call_long = w_call.*(strikes > F0);
    % pes_call = pes_call_long(pes_call_long~=0);

    % w = [pes_call;pes_put];
    % ******************************************************************

    % Compute the RMSE
    % rmse_vett(ii) = rmse( [OTM_call_model; OTM_put_model], ...
    %    [OTM_call_market; OTM_put_market], W = w );

    rmse_vett(ii) = rmse( [OTM_put_model; OTM_call_model], ...
    [OTM_put_market; OTM_call_market]);
        
end

% Compute the total RMSE
rmse_tot = sum(weights.*rmse_vett);

% rmse_tot = sum(rmse_vett);

end