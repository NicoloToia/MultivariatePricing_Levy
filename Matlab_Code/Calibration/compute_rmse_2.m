function rmse_tot = compute_rmse_2(Market, TTM, sigma, kappa, theta, M, dz, flag)
% This function computes the root mean squared error (RMSE) between the model and the market prices for each
% maturity and each strike. The RMSE is computed considering only the prices that are outside the bid-ask spread.
%
% INPUTS 
% Market: struct containing the market data
% TTM: vector of expiries
% sigma: volatility
% kappa: volatility of the volatility
% theta: skewness of the volatility
% M: N = 2^M is the number of points in the grid
% dz: grid spacing
% flag: model selection NIG or VG
%
% OUTPUTS
%
% rmse_tot: total RMSE (root mean squared error) between the model and the market prices

% Initialize rmse vector
rmse_vett = zeros(length(TTM), 1);

% Compute weights to overweight short maturities errors on prices
weights = flip((TTM./TTM(end))/sum(TTM./TTM(end)));

% Cycle over expiries
for ii = 1:min(length(TTM),20)

    % Import data from the Market struct
    F0 = Market.F0(ii).value;
    strikes = [Market.strikes(ii).value]';
    B0 = Market.B_bar(ii).value;
    put = Market.midPut(ii).value;
    call = Market.midCall(ii).value;
    callAsk = Market.callAsk(ii).prices;
    callBid = Market.callBid(ii).prices;
    putAsk = Market.putAsk(ii).prices;
    putBid = Market.putBid(ii).prices;    

    % Compute the log-moneyness
    log_moneyness = log(F0./strikes);

    % Compute the call prices via Lewis formula
    callPrices = callIntegral(B0, F0, sigma, kappa, theta, TTM(ii), log_moneyness, M, dz, flag);

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
    OTM_call_market_ask = callAsk(OTM_call_index:end);
    OTM_call_market_bid = callBid(OTM_call_index:end);

    % Put prices for OTM options
    OTM_put_market = put(1:OTM_put_index);
    OTM_put_market_ask = putAsk(1:OTM_put_index);
    OTM_put_market_bid = putBid(1:OTM_put_index);

    % check if the model price is inside the bid-ask spread
    % if so remove the point from the RMSE computation
    % do it if the vector has no nan values

    if ~any(isnan(OTM_call_model)) && ~any(isnan(OTM_put_model))

        OTM_call_model_f = OTM_call_model(OTM_call_model > OTM_call_market_ask' | OTM_call_model < OTM_call_market_bid');
        OTM_put_model_f = OTM_put_model(OTM_put_model > OTM_put_market_ask' | OTM_put_model < OTM_put_market_bid');
        OTM_call_market_f = OTM_call_market(OTM_call_model > OTM_call_market_ask' | OTM_call_model < OTM_call_market_bid');
        OTM_put_market_f = OTM_put_market(OTM_put_model > OTM_put_market_ask' | OTM_put_model < OTM_put_market_bid');

    else
        continue
    end

    % Compute the RMSE
   rmse_vett(ii) = rmse( [OTM_put_model_f; OTM_call_model_f], ...
    [OTM_put_market_f; OTM_call_market_f]);
        
end

% Compute the total RMSE
rmse_tot = sum(weights.*rmse_vett);

end