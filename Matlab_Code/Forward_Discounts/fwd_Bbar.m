function Market = fwd_Bbar(Market)
% This function returns the original struct adding new information in addition 
% to the original data: Market discount factors, Forward prices, Bid and Ask
%
% INPUT
%
% Market: Original struct with Market data
%
% OUTPUT
%
% Market: Modified struct with additional data
%           - B_bar: Market discount factors
%           - F: Forward prices
%           - FBid: Bid Forward prices
%           - FAsk: Ask Forward prices
%           - F0: Average Forward prices for each maturity

% We loop over the maturities
for ii = 1:length(Market.datesExpiry)
        
    % We compute the synthetic forward: G_bid(K), G_ask(K), G(K)
    GBid = Market.callBid(ii).prices - Market.putAsk(ii).prices;
    GAsk = Market.callAsk(ii).prices - Market.putBid(ii).prices;
    G = (GBid + GAsk)/2;

    % We compute G_hat and K_hat (sample mean of G(K) and K's)
    G_hat = mean(G);
    K_hat = mean(Market.strikes(ii).value);

    % We compute market discount factors between t0 and T_i
    Num = sum((Market.strikes(ii).value - K_hat).*(G - G_hat));
    Den = sum((Market.strikes(ii).value - K_hat).^2);
    Market.B_bar(ii).value = - Num/Den;

    % We compute forward price F, F_bid and F_ask
    Market.F(ii).value = G./Market.B_bar(ii).value + Market.strikes(ii).value;
    Market.FBid(ii).value = GBid./Market.B_bar(ii).value + Market.strikes(ii).value;
    Market.FAsk(ii).value = GAsk./Market.B_bar(ii).value + Market.strikes(ii).value;
    
    % Compute the average of forward prices for each maturity: F0
    Market.F0(ii).value = mean(Market.F(ii).value);

    % Compute the mid prices of the options
    Market.midCall(ii).value = (([Market.callBid(ii).prices] + [Market.callAsk(ii).prices])/2)';
    Market.midPut(ii).value = (([Market.putBid(ii).prices] + [Market.putAsk(ii).prices])/2)';

end

end