function plot_model_ImpVol(Model, Market, plotTitle)
% This function plots the market implied volatilities vs the model implied volatilities
%
% INPUTS
% Model: structure containing the model implied volatilities
% Market: structure containing the market implied volatilities
% plotTitle: title of the plot

% cycle through the different maturities
for ii = 1:length(Market.datesExpiry)
    figure;
    plot(Market.strikes(ii).value, Model.OTM_ImpVol(ii).value, 'b', 'LineWidth', 1);
    hold on;
    plot(Market.strikes(ii).value, Market.OTM_ImpVol(ii).value, 'r', 'LineWidth', 1);
    title([plotTitle, ' at maturity: ', num2str(ii)]);
    xlabel('Strikes');
    ylabel('Implied Volatilities');
    legend('Model ImpVol', 'Market ImpVol', 'Location', 'best');
    grid on;
    set(gca, 'LineWidth', 1.5, 'FontSize', 12);
    set(gcf, 'Position', [100, 100, 800, 600]);
    hold off;
end

end
