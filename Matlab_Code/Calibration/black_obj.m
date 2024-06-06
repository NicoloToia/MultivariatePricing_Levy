function obj = black_obj(Market, maturity, sigma)
% This function defines the objective function for the calibration of the 2-dimensional process using the Black model
%
% INPUTS
%  Market: structure containing the market data
%  maturity: vector of maturities
%  sigma: volatility
%
% OUTPUT
%  obj: objective function value

% Initialize the RMSE (root mean square error) vector
rmse_vett = zeros(length(maturity),1);

% Cycle over the maturities
for ii = 1:length(maturity)

    % EU market data from the structure
    F0 = Market.F0(ii).value; % forward price
    strikes = Market.strikes(ii).value'; % strikes
    B0 = Market.B_bar(ii).value; % discount factor
    put = Market.midPut(ii).value; % mid prices put
    call = Market.midCall(ii).value; % mid prices call

    % Compute the market zero rate
    rate = -log(B0)/maturity(ii);

    % Compute call prices via black
    [callPrices, putPrices] = blkprice(F0, strikes, rate, maturity(ii), sigma);
   
    % Extract the model prices for calls and puts
    % Find indexes
    OTM_put_index = sum((strikes <= F0) == 1);
    OTM_call_index = OTM_put_index + 1;
    
    % Call prices for OTM options
    OTM_call_model = callPrices(OTM_call_index:end);

    % Put prices for OTM options
    OTM_put_model = putPrices(1:OTM_put_index);
    
    % Extract the market prices for calls and puts
    % Call prices for OTM options
    OTM_call_market = call(OTM_call_index:end);

    % Put prices for OTM options
    OTM_put_market = put(1:OTM_put_index);

    % Compute the RMSE
    rmse_vett(ii) = rmse( [OTM_call_model; OTM_put_model], ...
       [OTM_call_market; OTM_put_market] );

end

% Objective function
obj = sum(rmse_vett);

end