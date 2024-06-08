function zGrid_interpolated = interpolateGrid(y_strikes, zGrid, numExpiries)
    zGrid_interpolated = zGrid;
    for ii = 1:numExpiries
        vector_numbers = ~isnan(zGrid(:, ii));
        indexes = find(vector_numbers > 0);
        if ~isempty(indexes)
            zGrid_interpolated(indexes(1):indexes(end), ii) = interp1(y_strikes(indexes), zGrid(indexes, ii), y_strikes(indexes(1):indexes(end)));
        end
    end
    zGrid_interpolated(zGrid_interpolated == 0) = NaN;
end