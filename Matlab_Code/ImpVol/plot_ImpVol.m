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
end

% Figure improvements
grid on;
title(plotTitle);
xlabel('Strike');
ylabel('Implied Volatility');
    
end