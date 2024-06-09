function rmse_tot = compute_rmse(Market, TTM, sigma, kappa, theta, M, dz, flag, flag_timeWindow)
% This function computes the root mean squared error (RMSE) between the model and the market prices for each
% maturity and each strike. The calibation is performed considering both markets.
%
% INPUTS 
% Market: struct containing the market data
% TTM: vector of expiries
% sigma: volatility
% kappa: volatility of the volatility
% theta: skewness of the volatility
% M: N = 2^M is the number of points in the grid
% dz: grid spacing
% flag: model selection: NIG or VG
%
% OUTPUTS
%
% rmse_tot: total RMSE (root mean squared error) between the model and the market prices

% Initialize rmse vector
rmse_vett = zeros(length(TTM), 1);


% LINEAR weights not used in the script, but described in the report
% weights = flip((TTM./TTM(end))/sum(TTM./TTM(end)));

% TRIANGULAR weights not used in the script, but described in the report
% settlement = datenum("07/09/2023");
% targetDate = datetime(settlement, 'ConvertFrom', 'datenum') + calyears(1);
% targetDate(~isbusday(targetDate, eurCalendar())) = busdate(targetDate(~isbusday(targetDate, eurCalendar())), 'modifiedfollow', eurCalendar());
% targetDate = datenum(targetDate);


% % Calculate the difference in days from the target date
% daysDiff = abs(datenum(Market.datesExpiry) - (targetDate));

% % Convert the differences to weights using an exponential decay function
% decayRate = 0.01; % Adjust decay rate as needed
% weights = exp(-decayRate * daysDiff);

% % Normalize the weights
% weights = (weights' / sum(weights));

% Cycle over expiries

if flag_timeWindow == 1
    date_calib = 739424;
    % find the index of the date_calib in the time to maturity vector
    index_calib = find(datenum(Market.datesExpiry) == date_calib);
    vect = index_calib-2:index_calib+2;
else
    vect = 1:min(length(TTM),19);
end

for ii = vect

    % Import data from the Market struct
    F0 = Market.F0(ii).value;
    strikes = [Market.strikes(ii).value]';
    B0 = Market.B_bar(ii).value;
    put = Market.midPut(ii).value;
    call = Market.midCall(ii).value;

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

    % Put prices for OTM options
    OTM_put_market = put(1:OTM_put_index);

    % %volumes weights, not used in the script, but described in the report
    % % ******************************************************************
    % w_call = Market.Volume_call(ii).volume; 
    % w_put = Market.Volume_put(ii).volume;

    % % Pesi
    % pes_put_long = w_put.*(strikes <= F0);
    % pes_put = pes_put_long(pes_put_long~=0);
    % pes_call_long = w_call.*(strikes > F0);
    % pes_call = pes_call_long(pes_call_long~=0);

    % w = [pes_call;pes_put];
    % % ******************************************************************

    % Compute the RMSE

    % uncomment to use the volumes weights
    % rmse_vett(ii) = rmse( [OTM_call_model; OTM_put_model], ...
    %    [OTM_call_market; OTM_put_market], W = w );

    % comment to use the volumes weights
    rmse_vett(ii) = rmse( [OTM_put_model; OTM_call_model], ...
    [OTM_put_market; OTM_call_market]);
        
end

% Compute the total RMSE

% uncomment to use time weights
% rmse_tot = sum(weights.*rmse_vett);

% % comment to use time weights
rmse_tot = sum(rmse_vett);

end