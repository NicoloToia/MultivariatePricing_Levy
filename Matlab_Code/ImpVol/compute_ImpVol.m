function Market = compute_ImpVol(Market, TTM, rates)
% This function computes the implied volatilities for real-world Market & Market model data
%
% INPUTS
%
% Market: structure with the market data
% TTM:    time to maturity in year fractions
% rates:  market zero rates
%
% OUTPUTS
%
% Market: structure with the market data and the implied volatilities

% Cycle over the different maturities
for ii = 1:length(Market.datesExpiry)

    % Compute the implied volatilities via the Black formula (vectorial implementation for the strikes)
    Market.ImpVol_call(ii).value = blkimpv(Market.F0(ii).value, [Market.strikes(ii).value]',rates(ii),...
                TTM(ii), [Market.midCall(ii).value], 'Class', {'Call'});
    
    Market.ImpVol_put(ii).value = blkimpv(Market.F0(ii).value, [Market.strikes(ii).value]',rates(ii),...
                TTM(ii), [Market.midPut(ii).value], 'Class', {'Put'});

    % % implied vol for ask and bid prices
    % Market.ImpVol_call(ii).ask = blkimpv(Market.F0(ii).value, [Market.strikes(ii).value]',rates(ii),...
    %             TTM(ii), [Market.callAsk(ii).prices]', 'Class', {'Call'});
    % 
    % Market.ImpVol_call(ii).bid = blkimpv(Market.F0(ii).value, [Market.strikes(ii).value]',rates(ii),...
    %             TTM(ii), [Market.callBid(ii).prices]', 'Class', {'Call'});
    % 
    % Market.ImpVol_put(ii).ask = blkimpv(Market.F0(ii).value, [Market.strikes(ii).value]',rates(ii),...
    %             TTM(ii), [Market.putAsk(ii).prices]', 'Class', {'Put'});
    % 
    % Market.ImpVol_put(ii).bid = blkimpv(Market.F0(ii).value, [Market.strikes(ii).value]',rates(ii),...
    %             TTM(ii), [Market.putBid(ii).prices]', 'Class', {'Put'});

end

end