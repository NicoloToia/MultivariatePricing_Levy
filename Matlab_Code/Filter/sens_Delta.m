function Market = sens_Delta(Market, TTM, rates)
% This function computes the delta sensitivity for the market data
%
% INPUTS:
% Market: structure with the market data
% TTM: time to maturity in year fractions
% rates: market zero rates
%
% OUTPUT:
% Market: structure with the market data and the delta sensitivity

% Call the needed variables from the struct
S0 = Market.spot;
% Null dividend
d = 0;

% Cycle over the different expiries and for each of them compute the delta sensitivities
for ii = 1:length(TTM)
    % Compute the delta sensitivities for the i-th maturity and store them in the struct
    [Market.sensDelta.call(ii).value, Market.sensDelta.put(ii).value] = ...
        blsdelta(S0, [Market.strikes(ii).value]', rates(ii), TTM(ii), [Market.OTM_ImpVol(ii).value], d);
end

end