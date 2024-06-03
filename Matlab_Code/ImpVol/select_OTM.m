function Market = select_OTM(Market)
% This function selects the out of the money options from the market data
% and build a new implied volatility smile
%
% INPUTS
% Market: structure with the market data
%
% OUTPUTS
% Market: structure with the market data and the OTM implied volatilities

% Call the needed variables from the struct
F0 = [Market.F0.value];

% Cycle over the different expiries and for each of them build the OTM
% The smile is constructed considering only the OTM options:
%   1. If F0 >= K, use put options
%   2. If F0 < K, use call options

for ii = 1:length(Market.datesExpiry)

    %volume = zeros(length(Market.strikes(ii).value),1);

    % Find the index of the strike before the forward price
    idx = find(Market.strikes(ii).value < F0(ii), 1, 'last');

    % Create the smile for the i-th maturity
    Market.OTM_ImpVol_put(ii).value = Market.ImpVol_put(ii).value(1:idx);
    Market.OTM_ImpVol_call(ii).value = Market.ImpVol_call(ii).value(idx+1:end);
    Market.OTM_ImpVol(ii).value = [Market.OTM_ImpVol_put(ii).value; Market.OTM_ImpVol_call(ii).value];
    
    % Check for NaN values in the implied volatilities
    if isnan(Market.OTM_ImpVol(ii).value)      
        error('NaN values in the implied volatilities');
    end

end

end