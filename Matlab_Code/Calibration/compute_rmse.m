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

weights = flip((TTM./TTM(end))/sum(TTM./TTM(end)));


% Cycle over expiries
for ii = 1:length(TTM)



    % Import data from the Market struct
    F0 = Market.F0(ii).value;
    strikes = Market.strikes(ii).value;
    B0 = Market.B_bar(ii).value;
    put = Market.midPut(ii).value';
    call = Market.midCall(ii).value';

    % Compute the log-moneyness
    log_moneyness = log(F0./strikes);

    % Compute the call prices via Lewis formula
    callPrices = callIntegral(B0, F0, alpha, sigma, kappa, theta, TTM(ii), log_moneyness, M, dz, 'FFT');

    % Compute the put prices via put-call parity
    putPrices = callPrices - B0*(F0 - strikes);

    % Compute the RMSE
    % N = length(strikes);
    % rmse(ii) = sqrt( sum( ((putPrices - put).*(strikes <= F0)).^2 ) / (N*sum(strikes <= F0)) ) ...
    %                 + sqrt( sum( ((callPrices - call).*(strikes > F0)).^2) / (N*sum(strikes > F0)) );

    % rmse_vett(ii) = rmse( [callPrices.*(strikes <= F0), putPrices.*(strikes > F0)], ...
    %    [call.*(strikes <= F0), put.*(strikes > F0)] );
    jolly = callPrices.*(strikes <= F0);
    prezzi_call = jolly(jolly~=0);
    jolly = putPrices.*(strikes > F0);
    prezzi_put = jolly(jolly~=0);

    jolly = call.*(strikes <= F0);
    call = jolly(jolly~=0);
    jolly = put.*(strikes > F0);
    put = jolly(jolly~=0);

    rmse_vett(ii) = rmse( [prezzi_call, prezzi_put], ...
       [call, put]);
        
end

% Compute the total RMSE
rmse_tot = sum(weights.*rmse_vett);

end