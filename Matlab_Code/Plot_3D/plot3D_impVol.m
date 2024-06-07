function plot3D_impVol(Market)

% find the boundaries for the y and the smallest step 
y_min = min(Market.strikes(1).value);
y_max = max(Market.strikes(1).value);
step_min = min(Market.strikes(1).value(2:end)-Market.strikes(1).value(1:end-1));
for ii = 1 : length(Market.datesExpiry)
    % find the boundaries for the y
    y_min = min([Market.strikes(ii).value, y_min]);
    y_max = max([Market.strikes(ii).value, y_max]);

    % find the smallest step 
    step_min = min([Market.strikes(ii).value(2:end)-Market.strikes(ii).value(1:end-1), step_min]);
end

y_strikes = (y_min : step_min : y_max)';

x_expiries = Market.datesExpiry';

[xGrid_new, yGrid_new] = meshgrid(x_expiries, y_strikes);

zGrid_new = NaN(length(y_strikes), length(x_expiries));

for ii = 1 : length(x_expiries)
    for kk = 1 : length(y_strikes)
        for jj = 1 : length(Market.strikes(ii).value)
            index = max(find(Market.strikes(ii).value == y_strikes(kk),1),0);
            if index > 0 
                zGrid_new(kk, ii) = Market.OTM_ImpVol(ii).value(index);
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

s = surf(xGrid_new, yGrid_new, zGrid_new_interpolated);
get(s);
set(s, 'EdgeAlpha',0.1);
xlabel('Expiries');
ylabel('Strikes');
zlabel('Implied Volatilities');
title('3D Surf of Implied Volatilities');
grid on;