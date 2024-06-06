function Market_filtered = flatten_smile(Market_filtered, target_maturity, flag)
% This function flattens the smile
%
% Inputs
% Market_filtered:


% flag : set the flag to 1 if additional flattening is needed


% find the maturity given the target maturity between the available maturities in the structure
maturities = datenum(Market_filtered.datesExpiry);
[~, idx] = min(abs(maturities - target_maturity));

% if flag == 1:
% remove the strikes that make the smile not flat, also in the case of similar implied volatilities for different strikes
% Starting from the lower strike if the implied volatility (times a safety coefficent of 1.0025) is higher than the
% previous one we remove the strike and the corresponding call and put prices and implied volatility

if flag == 1
    
    implied_vol = Market_filtered.OTM_ImpVol(idx).value;
    strikes = Market_filtered.strikes(idx).value;
    implied_vol_shifted = implied_vol*1.0025;

    for ii = 1:length(strikes) - 1
        if implied_vol_shifted(ii + 1) > implied_vol(ii)
            % remove the strike
            Market_filtered.strikes(idx).value(ii) = [];
            
            % remove the call prices
            Market_filtered.callBid(idx).prices(ii) = [];
            Market_filtered.callAsk(idx).prices(ii) = [];
            Market_filtered.midCall(idx).value(ii) = [];


            % remove the put prices
            Market_filtered.putBid(idx).prices(ii) = [];
            Market_filtered.putAsk(idx).prices(ii) = [];
            Market_filtered.midPut(idx).value(ii) = [];
            
            % remove volumes
            Market_filtered.Volume_call(idx).volume(ii) = [];
            Market_filtered.Volume_put(idx).volume(ii) = [];

            % remove the implied volatility
            Market_filtered.OTM_ImpVol(idx).value(ii) = [];
        end
    end
end

% Moreover,
% remove the strikes that make the smile not flat (if still present)
% Starting from the lower strike if the implied volatility is higher than the
% previous one we remove the strike and the corresponding call and put prices and implied volatility

% update the structure
implied_vol = Market_filtered.OTM_ImpVol(idx).value;
strikes = Market_filtered.strikes(idx).value;


for ii = 1:length(strikes) - 1
    if implied_vol(ii + 1) > implied_vol(ii)
        % remove the strike
        Market_filtered.strikes(idx).value(ii) = [];
        
        % remove the call prices
        Market_filtered.callBid(idx).prices(ii) = [];
        Market_filtered.callAsk(idx).prices(ii) = [];
        Market_filtered.midCall(idx).value(ii) = [];
        
        % remove the put prices
        Market_filtered.putBid(idx).prices(ii) = [];
        Market_filtered.putAsk(idx).prices(ii) = [];
        Market_filtered.midPut(idx).value(ii) = [];

        % remove volumes
        Market_filtered.Volume_call(idx).volume(ii) = [];
        Market_filtered.Volume_put(idx).volume(ii) = [];
        
        % remove the implied volatility
        Market_filtered.OTM_ImpVol(idx).value(ii) = [];
    end
end



end