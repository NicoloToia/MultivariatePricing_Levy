function plot_returns(Market_EU, Market_US, Returns)
% This function plots the returns of the EURO STOXX50 and S&P500 indexes
%
% INPUT
% Returns: struct containing the daily and yearly returns of the EURO STOXX50 and S&P500 indexes

% Yearly returns linspace
x_yearly = linspace(1,length(Returns.Daily(:,1)),length(Returns.Annually(:,1))+1);
% Daily returns linspace
x_daily = 1:length(Returns.Daily(:,1))+1;

% Plot the returns
figure;
% Plot the yearly returns EU and US
plot(x_yearly,flip(Market_EU.spot.*ret2price(Returns.Annually(:,2))),  '--', 'Color', [0.3 0.75 0.93], 'LineWidth', 2);
hold on;
plot(x_yearly,flip(Market_US.spot.*ret2price(Returns.Annually(:,1))), '--', 'Color', [1 0.6 0.78], 'LineWidth', 2);
% Plot the daily returns EU and US
plot(x_daily,flip(Market_EU.spot.*ret2price(Returns.Daily(:,2))), 'b', 'LineWidth', 2);
plot(x_daily,flip(Market_US.spot.*ret2price(Returns.Daily(:,1))), 'r', 'LineWidth', 2); 

% Improve the plot
title('Returns trend', 'FontSize', 14);
xlabel('Tempo', 'FontSize', 12);
ylabel('Prezzo', 'FontSize', 12);
grid on;
legend('Yearly EURO STOXX50', 'Yearly S&P500', 'Daily EURO STOXX50', 'Daily S&P500', 'Location', 'southeast');
set(gca, 'LineWidth', 1.5, 'FontSize', 12);
set(gcf, 'Position', [100, 100, 800, 600]);
xticks(linspace(1,length(Returns.Daily(:,1)),length(Returns.Annually(:,1))));
xticklabels({'-13y', '-12y', '-11y', '-10y', '-9y', '-8y', '-7y', '-6y', '-5y', '-4y', '-3y', '-2y', '-1y', 'Today'});
hold off;

end