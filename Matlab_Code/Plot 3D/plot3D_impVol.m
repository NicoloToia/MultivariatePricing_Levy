function plot3D_impVol(Market)

    % 3D plot
    % Step 1: Set the grid
    
    % find the boundaries for the y (strikes)
    y_min = min(Market.strikes(1).value);
    y_max = max(Market.strikes(1).value);

    for ii = 1 : length(Market.datesExpiry)
        y_min = min([Market.strikes(ii).value, y_min]);
        y_max = max([Market.strikes(ii).value, y_max]);
    end
    
    [xGrid, yGrid, zGrid] = deal([]);
    
    for ii = 1:length(Market.datesExpiry)
        expiry = Market.datesExpiry(ii);
        strikes = Market.strikes(ii).value;
        impliedVols = Market.OTM_ImpVol(ii).value;
        
    
        [x, y] = meshgrid(expiry, strikes);
        
        xGrid = [xGrid; x(:)];
        yGrid = [yGrid; y(:)];
        zGrid = [zGrid; impliedVols(:)];
    end
    
    % Step 2: Crete the plot
    scatter3(xGrid, yGrid, zGrid);
    xlabel('Expiries');
    ylabel('Strikes');
    zlabel('Implied Volatilities');
    title('3D Scatter Plot of Implied Volatilities');
    grid on;
end