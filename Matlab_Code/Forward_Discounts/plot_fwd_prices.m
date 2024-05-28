function plot_fwd_prices(Market)
% This function plots forward price with respect to strikes for each maturity
% that appears in the market data struct (mid, Bid, Ask)
%
% INPUT
%
% Market: Struct with market data

% We loop over the maturities
for ii = 1:length(Market.datesExpiry)
    % Inizialize the figure
    figure;
    
    % We plot the forward prices (mid, Bid, Ask) for each maturity
    plot(Market.strikes(ii).value, Market.F(ii).value, 'LineWidth', 2, 'Marker', 'o');
    hold on;
    plot(Market.strikes(ii).value, Market.FBid(ii).value, 'LineWidth', 2, 'Marker', 'x');
    plot(Market.strikes(ii).value, Market.FAsk(ii).value, 'LineWidth', 2, 'Marker', 's');
    
    % Figure improvements
    title(['Forward Prices w.r.t. Strikes for Expiry Date: ', datestr(Market.datesExpiry(ii))]);
    xlabel('Strikes');
    ylabel('Fwd Prices');
    grid on;
    legend({'F', 'FBid', 'FAsk'}, 'Location', 'Best');
    set(gca, 'FontSize', 14);
    set(gcf, 'Position', [100, 100, 800, 600]); 
    hold off;

end

end