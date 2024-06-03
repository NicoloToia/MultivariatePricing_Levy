function price = levy_pricing(Market_US, Market_EU, settlement, targetDate, ...
                                alpha, kappa_US, kappa_EU, sigma_US, sigma_EU, theta_US, theta_EU, N_sim)
% This function computes the price of a derivative using a Monte Carlo simulation under Lévy processes
% 
% INPUTS
%  Market_US: structure containing the market data for the US market
%  S0_EU: spot price of the European market
%  settlement: settlement date
%  targetDate: target date
%  alpha: Lévy exponent
%  kappa_US: Lévy parameter for the US market
%  kappa_EU: Lévy parameter for the EU market
%  sigma_US: volatility for the US market
%  sigma_EU: volatility for the EU market
%  theta_US: drift for the US market
%  theta_EU: drift for the EU market
%  N_sim: number of simulations
%
% OUTPUT
%  price: price of the derivative

% Recall variables from the market data
% Import the spot prices
S0_US = Market_US.spot;
S0_EU = Market_EU.spot;

% Import the expiries
Expiries_US = datenum([Market_US.datesExpiry]');
Expiries_EU = datenum([Market_EU.datesExpiry]');

% Import the market discounts B_bar
B_bar_US = [Market_US.B_bar.value]';
B_bar_EU = [Market_EU.B_bar.value]';

% Compute the discount via interpolation
discount_US = intExtDF(B_bar_US, Expiries_US, targetDate);
discount_EU = intExtDF(B_bar_EU, Expiries_EU, targetDate);

% Compute the forward prices
F0_US = S0_US/discount_US;
F0_EU = S0_EU/discount_EU;

% Compute the time to maturity
ACT_365 = 3;
ttm = yearfrac(settlement, targetDate, ACT_365);

% Simulation of the NIG processes
% Use a Montecarlo simulation to compute the call prices
% draw the standard normal random variables
g = mvnrnd([0; 0], [1 0.801; 0.801 1], N_sim);

% draw the inverse gaussian random variables
G_EU = random('inversegaussian', 1, ttm/kappa_EU, N_sim, 1);
G_US = random('inversegaussian', 1, ttm/kappa_US, N_sim, 1);

ft_EU = sqrt(ttm) * sigma_EU * sqrt(G_EU) .* g(:,1) - (0.5 + theta_EU) * ttm * sigma_EU^2 * G_EU...
            - ttm./kappa_EU * (1-sqrt(1-2.*kappa_EU.*theta_EU - kappa_EU.*sigma_EU .^2));

ft_US = sqrt(ttm) * sigma_US * sqrt(G_US) .* g(:,2) - (0.5 + theta_US) * ttm * sigma_US^2 * G_US...
            - ttm./kappa_US * (1-sqrt(1-2.*kappa_US.*theta_US - kappa_US.*sigma_US .^2));

S1_US = F0_US * exp(ft_US); 

S1_EU = F0_EU * exp(ft_EU); 

% indicator function US
Indicator_function = (S1_EU < 0.95 * S0_EU);

% Compute the payoff
payoff = max(S1_US - S0_US, 0) .* Indicator_function;

% Compute the price
price = discount_US * mean(payoff);

% Confidence interval
a = 0.01;
CI = norminv(1-a)*std(payoff)/sqrt(N_sim);
priceCI = [price - CI, price + CI];

% Display the results
fprintf('------------------------------------------------------------------\n');
fprintf('The price of the derivative is: %.4f\n', price);
fprintf('The confidence interval is: [%.4f, %.4f]\n', priceCI(1), priceCI(2));
fprintf('------------------------------------------------------------------\n');

end