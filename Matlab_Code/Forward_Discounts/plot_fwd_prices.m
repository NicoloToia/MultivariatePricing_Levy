function plot_fwd_prices(Market, title_mkt)
% This function plots forward price with respect to strikes for each maturity
% that appears in the market data struct (mid, Bid, Ask)
%
% INPUT
%
% Market: Struct with market data

% We loop over the maturities
for ii = 1:length(Market.datesExpiry)
    % Initialize the figure
    figure;
    
    % We plot the forward prices (mid, Bid, Ask) for each maturity
    plot(Market.strikes(ii).value, Market.F(ii).value, 'g', 'LineWidth', 1);
    hold on;
    plot(Market.strikes(ii).value, Market.FBid(ii).value, 'b', 'LineWidth', 1.5, 'Marker', 'x');
    plot(Market.strikes(ii).value, Market.FAsk(ii).value, 'r', 'LineWidth', 1.5, 'Marker', 'x');
    
    % Set horizontal lines for the minimum of the ask prices and the maximum of the bid prices
    yline(min(Market.FAsk(ii).value), 'r--', 'LineWidth', 1);
    yline(max(Market.FBid(ii).value), 'b--', 'LineWidth', 1);
    
    % Figure improvements
    title(['Forward Prices w.r.t. Strikes for Expiry Date: ', datestr(Market.datesExpiry(ii)), ' - ', title_mkt]);
    xlabel('Strikes');
    ylabel('Fwd Prices');
    grid on;
    legend({'F', 'FBid', 'FAsk', 'Min Ask', 'Max Bid'}, 'Location', 'Best');
    set(gca, 'FontSize', 10);
    % set(gcf, 'Position', [100, 100, 800, 600]); 
    hold off;
end



end
    