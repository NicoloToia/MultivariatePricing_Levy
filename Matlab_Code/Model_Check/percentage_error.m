function [error_EU, error_US] = percentage_error(Model_EU, Model_US, Market_EU, Market_US)
% This function calculates the percentage error of the model calibrated prices for the EU
% and US markets with respect to the real prices.
%
% INPUTS
% Model_EU: structure containing the calibrated model for the European
% Model_US: structure containing the calibrated model for the American
% Market_EU: structure containing the European market data
% Market_US: structure containing the American market data
%
% OUTPUTS
% error_EU: percentage error for the European market
% error_US: percentage error for the American market

% Inizialize the error vector (auxiliary variable)
errCall = zeros(length(Market_EU.datesExpiry),1);
errPut = zeros(length(Market_EU.datesExpiry),1);

% Cycle over EU expiries
for ii = 1:length(Market_EU.datesExpiry)
    % Real prices
    real_price_call = Market_EU.midCall(ii).value;
    real_price_put = Market_EU.midPut(ii).value;
    % Model prices
    model_price_call = Model_EU.midCall(ii).value;
    model_price_put = Model_EU.midPut(ii).value;
    % Errors
    e_call = norm(model_price_call - real_price_call)./norm(real_price_call);
    e_put = norm(model_price_put - real_price_put)./norm(real_price_put);
    % Store the errors
    errCall(ii) = e_call*100;
    errPut(ii) = e_put*100;
end

% Calculate the percentage error for the European market
error_EU = mean([errCall; errPut]);

% Inizialize the error vector (auxiliary variable)
errCall = zeros(length(Market_US.datesExpiry),1);
errPut = zeros(length(Market_US.datesExpiry),1);

% Cycle over US expiries
for ii = 1:length(Market_US.datesExpiry)
    % Real prices
    real_price_call = Market_US.midCall(ii).value;
    real_price_put = Market_US.midPut(ii).value;
    % Model prices
    model_price_call = Model_US.midCall(ii).value;
    model_price_put = Model_US.midPut(ii).value;
    % Errors
    e_call = norm(model_price_call - real_price_call)./norm(real_price_call);
    e_put = norm(model_price_put - real_price_put)./norm(real_price_put);
    % Store the errors
    errCall(ii) = e_call*100;
    errPut(ii) = e_put*100;
end

% Calculate the percentage error for the American market
error_US = mean([errCall; errPut]);

end