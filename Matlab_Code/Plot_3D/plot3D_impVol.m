function plot3D_impVol(Market1, Market2, title_name)
    % This function plots the market and model implied volatilities in 3D
    %
    % INPUTS
    %  Market1: structure containing the first set of implied volatilities
    %  Market2: structure containing the second set of implied volatilities
    %  title_name: title of the plot

% find the boundaries for the y and the smallest step 
y_min = min(Market1.strikes(1).value);
y_max = max(Market1.strikes(1).value);
step_min = min(Market1.strikes(1).value(2:end)-Market1.strikes(1).value(1:end-1));
for ii = 1 : length(Market1.datesExpiry)
    % find the boundaries for the y
    y_min = min([Market1.strikes(ii).value, y_min]);
    y_max = max([Market1.strikes(ii).value, y_max]);

    % find the smallest step 
    step_min = min([Market1.strikes(ii).value(2:end)-Market1.strikes(ii).value(1:end-1), step_min]);
end

y_strikes = (y_min : step_min : y_max)';

x_expiries = Market1.datesExpiry';

[xGrid_new, yGrid_new] = meshgrid(x_expiries, y_strikes);

zGrid_new = NaN(length(y_strikes), length(x_expiries));

for ii = 1 : length(x_expiries)
    for kk = 1 : length(y_strikes)
        for jj = 1 : length(Market1.strikes(ii).value)
            index = max(find(Market1.strikes(ii).value == y_strikes(kk),1),0);
            if index > 0 
                zGrid_new(kk, ii) = Market1.OTM_ImpVol(ii).value(index);
            else 
                zGrid_new(kk, ii) = NaN;
            end
        end
    end
end
% size(xGrid_new)
% size(yGrid_new)
zGrid_new_interpolated1 = zGrid_new;
for ii = 1 : length(x_expiries)
    vector_numbers = ~isnan(zGrid_new(:, ii));
    vector_not_numbers = isnan(zGrid_new(:, ii));
    indexes = find(vector_numbers > 0);
    zGrid_new_interpolated1(indexes(1):indexes(end), ii) =  interp1(y_strikes(indexes), zGrid_new(indexes,ii), y_strikes(indexes(1):indexes(end)));
end
for ii = 1 : length(x_expiries)
    for jj = 1 : length(y_strikes)
          if zGrid_new_interpolated1(jj, ii) == 0
              zGrid_new_interpolated1(jj, ii) = NaN;
          end
    end
end


%---------------------------------------------------------------------------------------------------------
% find the boundaries for the y and the smallest step 
y_min = min(Market2.strikes(1).value);
y_max = max(Market2.strikes(1).value);
step_min = min(Market2.strikes(1).value(2:end)-Market2.strikes(1).value(1:end-1));
for ii = 1 : length(Market2.datesExpiry)
    % find the boundaries for the y
    y_min = min([Market2.strikes(ii).value, y_min]);
    y_max = max([Market2.strikes(ii).value, y_max]);

    % find the smallest step 
    step_min = min([Market2.strikes(ii).value(2:end)-Market2.strikes(ii).value(1:end-1), step_min]);
end

y_strikes = (y_min : step_min : y_max)';

x_expiries = Market2.datesExpiry';

[xGrid_new, yGrid_new] = meshgrid(x_expiries, y_strikes);

zGrid_new = NaN(length(y_strikes), length(x_expiries));

for ii = 1 : length(x_expiries)
    for kk = 1 : length(y_strikes)
        for jj = 1 : length(Market2.strikes(ii).value)
            index = max(find(Market2.strikes(ii).value == y_strikes(kk),1),0);
            if index > 0 
                zGrid_new(kk, ii) = Market2.OTM_ImpVol(ii).value(index);
            else 
                zGrid_new(kk, ii) = NaN;
            end
        end
    end
end
% size(xGrid_new)
% size(yGrid_new)
zGrid_new_interpolated2 = zGrid_new;
for ii = 1 : length(x_expiries)
    vector_numbers = ~isnan(zGrid_new(:, ii));
    vector_not_numbers = isnan(zGrid_new(:, ii));
    indexes = find(vector_numbers > 0);
    zGrid_new_interpolated2(indexes(1):indexes(end), ii) =  interp1(y_strikes(indexes), zGrid_new(indexes,ii), y_strikes(indexes(1):indexes(end)));
end
for ii = 1 : length(x_expiries)
    for jj = 1 : length(y_strikes)
          if zGrid_new_interpolated2(jj, ii) == 0
              zGrid_new_interpolated2(jj, ii) = NaN;
          end
    end
end
% ------------------------------------------------------------------------


% plot Market surface
s1 = surf(xGrid_new, yGrid_new, zGrid_new_interpolated1);
set(s1,'EdgeColor', 'black', 'FaceColor', '[0.8500 0.3250 0.0980]', 'EdgeAlpha',0.25);

hold on;

% plot model surface
s2 = surf(xGrid_new, yGrid_new, zGrid_new_interpolated2);
set(s2, 'EdgeColor', 'black', 'FaceColor', '[0 0.4470 0.7410]', 'EdgeAlpha',0.25);

legend('Market', 'Model');
xlabel('Expiries');
ylabel('Strikes');
zlabel('Implied Volatilities');
title(title_name);
grid on;