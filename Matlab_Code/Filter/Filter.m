function Filtered_Market = Filter(Market)
% This function filters the market data based on the delta sensitivities
% If the delta is between 0.1 and 0.9 for calls and -0.9 and -0.1 for puts, the option is kept
%
% INPUTS
% Market: structure containing the market data
%
% OUTPUTS
% Filtered_Market: structure containing the filtered market data


% Cycle through the expiries
for ii = 1:length(Market.datesExpiry)

    % Filter deltas for the options with delta between 0.1 and 0.9 for calls and -0.9 and -0.1 for puts
    Market.sensDelta.call(ii).value = Market.sensDelta.call(ii).value.*(Market.sensDelta.call(ii).value>=0.1) ...
                                        .*(Market.sensDelta.call(ii).value<=0.9);

    Market.sensDelta.put(ii).value = Market.sensDelta.put(ii).value.*(Market.sensDelta.put(ii).value>=-0.9) ...
                                        .*(Market.sensDelta.put(ii).value<=-0.1);

    % Construct the new filtered market data
    % Expiries
    Filtered_Market.datesExpiry(ii) = Market.datesExpiry(ii);
    % Call Bid and Ask
    Filtered_Market.callBid(ii).prices = Market.callBid(ii).prices(Market.sensDelta.call(ii).value ~=0);
    Filtered_Market.callAsk(ii).prices = Market.callAsk(ii).prices(Market.sensDelta.call(ii).value ~=0);
    % Put Bid and Ask
    Filtered_Market.putBid(ii).prices = Market.putBid(ii).prices(Market.sensDelta.put(ii).value ~=0);
    Filtered_Market.putAsk(ii).prices = Market.putAsk(ii).prices(Market.sensDelta.put(ii).value ~=0);
    % Strikes
    Filtered_Market.strikes(ii).value = Market.strikes(ii).value((Market.sensDelta.call(ii).value ~=0 & ...
                                         Market.sensDelta.put(ii).value ~=0));
    % Spot
    Filtered_Market.spot = Market.spot;
    % Volumes
    Filtered_Market.Volume_call(ii).volume = Market.Volume_call(ii).volume(Market.sensDelta.call(ii).value ~=0);
    Filtered_Market.Volume_put(ii).volume = Market.Volume_put(ii).volume(Market.sensDelta.put(ii).value ~=0);
    % Discounts
    Filtered_Market.B_bar(ii).value = Market.B_bar(ii).value;
    % F0
    Filtered_Market.F0(ii).value = Market.F0(ii).value;
    % mid prices
    Filtered_Market.midCall(ii).value = Market.midCall(ii).value(Market.sensDelta.call(ii).value ~=0);
    Filtered_Market.midPut(ii).value = Market.midPut(ii).value(Market.sensDelta.put(ii).value ~=0);
    % Implied volatilities
    Filtered_Market.OTM_ImpVol(ii).value = Market.OTM_ImpVol(ii).value(Market.sensDelta.call(ii).value ~=0); 
end

end