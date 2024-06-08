function [xGrid, yGrid, zGrid] = prepareGrid(Market)
    % Find the boundaries for the y and the smallest step
    y_min = min(Market.strikes(1).value);
    y_max = max(Market.strikes(1).value);
    step_min = min(Market.strikes(1).value(2:end) - Market.strikes(1).value(1:end-1));
    for ii = 1:length(Market.datesExpiry)
        % Find the boundaries for the y
        y_min = min([Market.strikes(ii).value, y_min]);
        y_max = max([Market.strikes(ii).value, y_max]);

        % Find the smallest step 
        step_min = min([Market.strikes(ii).value(2:end) - Market.strikes(ii).value(1:end-1), step_min]);
    end

    y_strikes = (y_min:step_min:y_max)';
    x_expiries = Market.datesExpiry';
    [xGrid, yGrid] = meshgrid(x_expiries, y_strikes);
    zGrid = NaN(length(y_strikes), length(x_expiries));

    for ii = 1:length(x_expiries)
        for kk = 1:length(y_strikes)
            for jj = 1:length(Market.strikes(ii).value)
                index = max(find(Market.strikes(ii).value == y_strikes(kk), 1), 0);
                if index > 0
                    zGrid(kk, ii) = Market.OTM_ImpVol(ii).value(index);
                else
                    zGrid(kk, ii) = NaN;
                end
            end
        end
    end
end

