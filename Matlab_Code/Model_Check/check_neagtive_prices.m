function check_neagtive_prices(Model_EU, Model_US)
% This function checks if there are negative prices in the calibrated model for the EU and US market
%
% INPUTS
% Model_EU: structure with the calibrated prices for the EU market
% Model_US: structure with the calibrated prices for the US market

% Cycle over EU maturities
for ii = 1:length(Model_EU.datesExpiry)

    % Check if there are negative prices for calls
    if sum(Model_EU.midCall(ii).value < 0) > 0
        disp(['Negative prices call for the EU market at maturity: ', num2str(ii)]);
    end

    % Check if there are negative prices for puts
    if sum(Model_EU.midPut(ii).value < 0) > 0
        disp(['Negative prices put for the EU market at maturity: ', num2str(ii)]);
    end
end

% Cycle over US maturities
for ii = 1:length(Model_US.datesExpiry)

    % Check if there are negative prices for calls
    if sum(Model_US.midCall(ii).value < 0) > 0
        disp(['Negative prices call for the US market at maturity: ', num2str(ii)]);
    end

    % Check if there are negative prices for puts
    if sum(Model_US.midPut(ii).value < 0) > 0
        disp(['Negative prices put for the US market at maturity: ', num2str(ii)]);
    end

end