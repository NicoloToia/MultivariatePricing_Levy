function derivativePrice_MC = levy_pricing(Market_EU, S0_US, settlement, targetDate, ...
                                alpha, kappa_EU, kappa_US, sigma_EU, sigma_US, theta_EU, theta_US, nSim)

% This function computes the price of a derivative using a Monte Carlo simulation under Lévy processes
% 
% INPUTS
%  Market_EU: structure containing the market data for the European market
%  S0_US: spot price of the US market
%  settlement: settlement date
%  targetDate: target date
%  alpha: Lévy exponent
%  kappa_EU: Lévy parameter for the EU market
%  kappa_US: Lévy parameter for the US market
%  sigma_EU: volatility for the EU market
%  sigma_US: volatility for the US market
%  theta_EU: drift for the EU market
%  theta_US: drift for the US market
%  nSim: number of simulations
%
% OUTPUT
%  derivativePrice_MC: price of the derivative

S0_EU = Market_EU.spot;
% Simulation of the NIG processes
% use a MonteCarlo simulation to compute the call prices
Expiries = datenum([Market_EU.datesExpiry]');

B_bar = [Market_EU.B_bar.value]';

% Compute the discount
discount = intExtDF(B_bar, Expiries, targetDate);

% Compute the forward prices
F0_EU = S0_EU/discount;
F0_US = S0_US/discount; % dscount sbagliato

% più o meno giusti ma come li otteniamo?????????
F0_EU = 4345;
F0_US = 4615;

% Compute the time to maturity
ACT_365 = 3;
ttm = yearfrac(settlement, targetDate, ACT_365);

% compute the Laplace exponent EU & US
ln_L_EU = @(omega_EU) ttm/kappa_EU * (1 - alpha)/alpha * ...
    (1 - (1 + (omega_EU .* kappa_EU * sigma_EU^2)/(1-alpha)).^alpha );

ln_L_US = @(omega_US) ttm/kappa_US * (1 - alpha)/alpha * ...
    (1 - (1 + (omega_US .* kappa_US * sigma_US^2)/(1-alpha)).^alpha );

% draw the standard normal random variables
g = mvnrnd([0; 0], [1 0.801; 0.801 1], nSim);
% g = randn(nSim, 1);
% draw the inverse gaussian random variables
G_US = random('inversegaussian', 1, ttm/kappa_US, nSim, 1);
G_EU = random('inversegaussian', 1, ttm/kappa_EU, nSim, 1);

% ft_EU = sqrt(ttm) * sigma_EU * sqrt(G) .* g(:,1) - (0.5 + theta_EU) * ttm * sigma_EU^2 * G - ln_L_EU(theta_EU);
% ft_US = sqrt(ttm) * sigma_US * sqrt(G) .* g(:,2) - (0.5 + theta_US) * ttm * sigma_US^2 * G - ln_L_US(theta_US);

% ft_EU = sqrt(ttm) * sigma_EU * sqrt(G) .* g(:,1) - (0.5 + theta_EU) * ttm * sigma_EU^2 * G + ttm./kappa_EU * (1-sqrt(1-2.*kappa_EU.*theta_EU - kappa_EU.*sigma_EU .^2)) - ln_L_EU(theta_EU);
% ft_US = sqrt(ttm) * sigma_US * sqrt(G) .* g(:,2) - (0.5 + theta_US) * ttm * sigma_US^2 * G + ttm./kappa_US * (1-sqrt(1-2.*kappa_US.*theta_US - kappa_US.*sigma_US .^2)) - ln_L_US(theta_US);
% compute F(t) EU & US

% ft_EU = sqrt(ttm) * sigma_EU * sqrt(G) .* g(:,1) - (0.5 + theta_EU) * ttm * sigma_EU^2 * G;
% ft_US = sqrt(ttm) * sigma_US * sqrt(G) .* g(:,2) - (0.5 + theta_US) * ttm * sigma_US^2 * G;

% ft_EU = sqrt(ttm) * sigma_EU * sqrt(G) .* g - (0.5 + theta_EU) * ttm * sigma_EU^2 * G;
% ft_US = sqrt(ttm) * sigma_US * sqrt(G) .* g - (0.5 + theta_US) * ttm * sigma_US^2 * G;

% prove del 30/05

% ft_EU = sqrt(ttm) * sigma_EU * sqrt(G_EU) .* g(:,1) - (0.5 + theta_EU) * ttm * sigma_EU^2 * G_EU - ln_L_EU(theta_EU)...
%             - ttm./kappa_EU * (1-sqrt(1-2.*kappa_EU.*theta_EU - kappa_EU.*sigma_EU .^2));

% ft_US = sqrt(ttm) * sigma_US * sqrt(G_US) .* g(:,2) - (0.5 + theta_US) * ttm * sigma_US^2 * G_US - ln_L_US(theta_US)...
%             - ttm./kappa_US * (1-sqrt(1-2.*kappa_US.*theta_US - kappa_US.*sigma_US .^2));

ft_EU = sqrt(ttm) * sigma_EU * sqrt(G_EU) .* g(:,1) - (0.5 + theta_EU) * ttm * sigma_EU^2 * G_EU - ln_L_EU(theta_EU)...
                + ttm./kappa_EU * ( sqrt(1 - 2*kappa_EU) - 1 );

ft_US = sqrt(ttm) * sigma_US * sqrt(G_US) .* g(:,2) - (0.5 + theta_US) * ttm * sigma_US^2 * G_US - ln_L_US(theta_US)...
                + ttm./kappa_US * ( sqrt(1 - 2*kappa_US) - 1 );

S1_EU = F0_EU * exp(ft_EU); 

S1_US = F0_US * exp(ft_US); 

% indicator function US
Indicator_function = (S1_US < 0.95 * S0_US);

% check
prova = S1_EU - S0_EU;
maggioridi0 = sum(prova>0);
maggioridi02 = sum(Indicator_function);
prova2 = max(S1_EU - S0_EU, 0);

% Compute the payoff
payoff = max(S1_EU - S0_EU, 0) .* Indicator_function;

% Compute the price
derivativePrice_MC = discount * mean(payoff);

% Confidence interval
a = 0.01;
CI = norminv(1-a)*std(payoff)/sqrt(nSim);
priceCI = [derivativePrice_MC - CI, derivativePrice_MC + CI];

% Display the results
fprintf('------------------------------------------------------------------\n');
fprintf('The price of the derivative is: %.4f\n', derivativePrice_MC);
fprintf('The confidence interval is: [%.4f, %.4f]\n', priceCI(1), priceCI(2));
fprintf('------------------------------------------------------------------\n');

end