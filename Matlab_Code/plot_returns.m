function plot_returns(Returns)
% This function plots the returns of the EURO STOXX50 and S&P500 indexes
%
% INPUT
% Returns: struct containing the daily and yearly returns of the EURO STOXX50 and S&P500 indexes


% Plot the returns
figure;
% Plot the yearly returns EU and US
plot(linspace(1,3188,14),flip(EU_Market.spot.*ret2price(Returns.Returns.Annually(:,2))),  '--', 'Color', [0.3 0.75 0.93], 'LineWidth', 2);
hold on;
plot(linspace(1,3188,14),flip(US_Market.spot.*ret2price(Returns.Returns.Annually(:,1))), '--', 'Color', [1 0.6 0.78], 'LineWidth', 2);
% Plot the daily returns EU and US
plot(1:3189,flip(EU_Market.spot.*ret2price(Returns.Returns.Daily(:,2))), 'b', 'LineWidth', 2);
plot(1:3189,flip(US_Market.spot.*ret2price(Returns.Returns.Daily(:,1))), 'r', 'LineWidth', 2); 

% Improve the plot
title('Returns trend', 'FontSize', 14);
xlabel('Tempo', 'FontSize', 12);
ylabel('Prezzo', 'FontSize', 12);
grid on;
legend('Yearly EURO STOXX50', 'Yearly S&P500', 'Daily EURO STOXX50', 'Daily S&P500', 'Location', 'southeast');
set(gca, 'LineWidth', 1.5, 'FontSize', 12);
set(gcf, 'Position', [100, 100, 800, 600]);
xticks(linspace(1, 3188, 14));
xticklabels({'-13y', '-12y', '-11y', '-10y', '-9y', '-8y', '-7y', '-6y', '-5y', '-4y', '-3y', '-2y', '-1y', 'Today'});
hold off;

end