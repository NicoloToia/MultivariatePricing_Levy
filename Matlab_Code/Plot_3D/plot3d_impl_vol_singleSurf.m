function plot3d_impl_vol_singleSurf(Filtered_Market, title_name)
% This function plots the implied volatilities in 3D
% Input: Filtered_Market - structure with the filtered market data
%        title_name - string with the title of the plot
% Output: 3D plot of the implied volatilities

% Step 1: Set the grid
% find the boundaries for the y (strikes)
y_min = min(Filtered_Market.strikes(1).value);
y_max = max(Filtered_Market.strikes(1).value);

for ii = 1 : length(Filtered_Market.datesExpiry)
    y_min = min([Filtered_Market.strikes(ii).value, y_min]);
    y_max = max([Filtered_Market.strikes(ii).value, y_max]);
end

[xGrid, yGrid, zGrid] = deal([]);

for ii = 1:length(Filtered_Market.datesExpiry)
    expiry = Filtered_Market.datesExpiry(ii);
    strikes = Filtered_Market.strikes(ii).value;
    impliedVols = Filtered_Market.OTM_ImpVol(ii).value;
    

    [x, y] = meshgrid(expiry, strikes);
    
    xGrid = [xGrid; x(:)];
    yGrid = [yGrid; y(:)];
    zGrid = [zGrid; impliedVols(:)];
end

% Step 2: Crete the  scatter plot
figure;
scatter3(xGrid, yGrid, zGrid);
xlabel('Expiries');
ylabel('Strikes');
zlabel('Implied Volatilities');
title(title_name);
grid on;

% Step 3: Create the 3D surface plot
figure;
% find the boundaries for the y and the smallest step 
y_min = min(Filtered_Market.strikes(1).value);
y_max = max(Filtered_Market.strikes(1).value);
step_min = min(Filtered_Market.strikes(1).value(2:end)-Filtered_Market.strikes(1).value(1:end-1));
for ii = 1 : length(Filtered_Market.datesExpiry)
    % find the boundaries for the y
    y_min = min([Filtered_Market.strikes(ii).value, y_min]);
    y_max = max([Filtered_Market.strikes(ii).value, y_max]);

    % find the smallest step 
    step_min = min([Filtered_Market.strikes(ii).value(2:end)-Filtered_Market.strikes(ii).value(1:end-1), step_min]);
end

% create the grid
y_strikes = (y_min : step_min : y_max)';

x_expiries = Filtered_Market.datesExpiry';

[xGrid_new, yGrid_new] = meshgrid(x_expiries, y_strikes);

zGrid_new = NaN(length(y_strikes), length(x_expiries));

for ii = 1 : length(x_expiries)
    for kk = 1 : length(y_strikes)
        for jj = 1 : length(Filtered_Market.strikes(ii).value)
            index = max(find(Filtered_Market.strikes(ii).value == y_strikes(kk),1),0);
            if index > 0 
                zGrid_new(kk, ii) = Filtered_Market.OTM_ImpVol(ii).value(index);
            else 
                zGrid_new(kk, ii) = NaN;
            end
        end
    end
end

% size(xGrid_new)
% size(yGrid_new)

zGrid_new_interpolated = zGrid_new;

for ii = 1 : length(x_expiries)
    vector_numbers = ~isnan(zGrid_new(:, ii));
    vector_not_numbers = isnan(zGrid_new(:, ii));
    indexes = find(vector_numbers > 0);
    zGrid_new_interpolated(indexes(1):indexes(end), ii) =  interp1(y_strikes(indexes), zGrid_new(indexes,ii), y_strikes(indexes(1):indexes(end)));
end
for ii = 1 : length(x_expiries)
    for jj = 1 : length(y_strikes)
            if zGrid_new_interpolated(jj, ii) == 0
                zGrid_new_interpolated(jj, ii) = NaN;
            end
    end
end

% Plot the surface
s = surf(xGrid_new, yGrid_new, zGrid_new_interpolated);
set(s, 'EdgeAlpha',0.1);
xlabel('Expiries');
ylabel('Strikes');
zlabel('Implied Volatilities');
title(title_name);
grid on;
    
end 