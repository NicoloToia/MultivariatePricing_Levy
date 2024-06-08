function plot_model_prices(Model, Market, plotTitle)
% This function plots the model prices against the bid and ask prices for the market data.
%
% INPUTS
% Model: structure containing the calibrated model prices
% Market: structure containing the market prices
% plotTitle: title of the plot

% Cycle through the maturities
for ii = 1:length(Market.datesExpiry)

    % Plot the call prices
    figure;
    plot(Market.strikes(ii).value, Model.midCall(ii).value, 'b', 'LineWidth', 1);
    hold on;
    plot(Market.strikes(ii).value, Market.callAsk(ii).prices, 'r', 'LineWidth', 1);
    plot(Market.strikes(ii).value, Market.callBid(ii).prices, 'g', 'LineWidth', 1);
    title([plotTitle, ' (CALL) ', char(Market.datesExpiry(ii))]);
    xlabel('Strikes');
    ylabel('Prices');
    legend('Model prices', 'Ask prices', 'Bid prices', 'Location', 'best');
    grid on;
    set(gca, 'LineWidth', 1.5, 'FontSize', 12);
    set(gcf, 'Position', [100, 100, 800, 600]);
    hold off;
end

% Cycle through the maturities
for ii = 1:length(Market.datesExpiry)

    % Plot the put prices
    figure;
    plot(Market.strikes(ii).value, Model.midPut(ii).value, 'b', 'LineWidth', 1);
    hold on;
    plot(Market.strikes(ii).value, Market.putAsk(ii).prices, 'r', 'LineWidth', 1);
    plot(Market.strikes(ii).value, Market.putBid(ii).prices, 'g', 'LineWidth', 1);
    title([plotTitle, ' ', ' (PUT) ', char(Market.datesExpiry(ii))]);
    xlabel('Strikes');
    ylabel('Prices');
    legend('Model prices', 'Ask prices', 'Bid prices', 'Location', 'best');
    grid on;
    set(gca, 'LineWidth', 1.5, 'FontSize', 12);
    set(gcf, 'Position', [100, 100, 800, 600]);
    hold off;

end