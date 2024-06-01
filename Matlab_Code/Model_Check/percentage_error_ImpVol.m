function [error_EU, error_US] = percentage_error_ImpVol(Model_EU, Model_US, Market_EU, Market_US)
% This function calculates the percentage error of the model calibrated implied volatilities for the EU
% and US markets with respect to the real implied volatilities.
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
err = zeros(length(Market_EU.datesExpiry),1);

% Cycle over EU expiries (OTM implied volatilities)
for ii = 1:length(Market_EU.datesExpiry)
    % Real implied volatilities
    real_IV = Market_EU.OTM_ImpVol(ii).value;
    % Model implied volatilities
    model_IV = Model_EU.OTM_ImpVol(ii).value;
    % errors
    e = norm(model_IV - real_IV)/norm(real_IV);
    % Store the errors
    err(ii) = e*100;
end

% Calculate the percentage error for the European market
error_EU = mean(err);

% Inizialize the error vector (auxiliary variable)
err = zeros(length(Market_US.datesExpiry),1);

% Cycle over US expiries (OTM implied volatilities)
for ii = 1:length(Market_US.datesExpiry)
    % Real implied volatilities
    real_IV = Market_US.OTM_ImpVol(ii).value;
    % Model implied volatilities
    model_IV = Model_US.OTM_ImpVol(ii).value;
    % errors
    e = norm(model_IV - real_IV)/norm(real_IV);
    % Store the errors
    err(ii) = e*100;
end

% Calculate the percentage error for the American market
error_US = mean(err);

end