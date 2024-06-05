function plot_ImpVol(Market, plotTitle)
% This function plots the implied volatility smile
%
% INPUTS
% Market: structure with the market data
% plotTitle: title of the plot

% Initialize the figure
figure;

% Cycle over the different expiries and plot the implied volatility smile
for ii = 1:length(Market.datesExpiry)
    
    % Plot the implied volatility smile for the i-th maturity
    % smile is first composed by put then call options (append the two vectors)
    plot(Market.strikes(ii).value, Market.OTM_ImpVol(ii).value);
    hold on;
    % % plot also the bid and ask implied volatilities
    % plot(Market.strikes(ii).value, Market.OTM_ImpVol(ii).ask, 'r');
    % plot(Market.strikes(ii).value, Market.OTM_ImpVol(ii).bid, 'r');
    % % improve the plot
    % legend('Mid', 'Ask/Bid');
    % % title with the date of the expiry
    % title(datestr(Market.datesExpiry(ii)));
    % % axis labels
    % xlabel('Strike');
    % ylabel('Implied Volatility');
    % grid on;
    % hold off;

end

% Figure improvements
grid on;
title(plotTitle);
xlabel('Strike');
ylabel('Implied Volatility');
    
end