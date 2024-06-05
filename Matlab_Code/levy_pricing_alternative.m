function price = levy_pricing_alternative(Market_US, Market_EU, settlement, targetDate, sigma_US, sigma_EU, kappa_US, kappa_EU,...
            theta_US, theta_EU, nu_US, a_US, a_EU, nu_EU, Beta_Z,gamma_Z,nu_Z, nSim)
% This function computes the price of a barrier option using the Levy pricing alternative
%
% INPUTS
% Market_US: struct containing the market data for the US market
% Market_EU: struct containing the market data for the EU market
% settlement: settlement date
% targetDate: target date
% sigma_US: volatility of the US market
% sigma_EU: volatility of the EU market
% kappa_US: volatility of the volatility of the US market
% kappa_EU: volatility of the volatility of the EU market
% theta_US: skewness of the volatility of the US market
% theta_EU: skewness of the volatility of the EU market
% nu_US: volatility of the jumps of the US market
% nu_EU: volatility of the jumps of the EU market
% nu_Z: volatility of the jumps of the common factor
% nSim: number of simulations
%
% OUTPUTS
% price: price of the option
% 
% param --> [a_US, a_EU, Beta_Z, gamma_Z]

Expiries_US = datenum([Market_US.datesExpiry]');
Expiries_EU = datenum([Market_EU.datesExpiry]');

B_bar_US = [Market_US.B_bar.value]';
B_bar_EU = [Market_EU.B_bar.value]';

% Compute the discount
discount_US = intExtDF(B_bar_US, Expiries_US, targetDate);
discount_EU = intExtDF(B_bar_EU, Expiries_EU, targetDate);

% Compute the time to maturity
ACT_365 = 3;
ttm = yearfrac(settlement, targetDate, ACT_365);

% compute rates at ttm
rate_US = -log(discount_US)/ttm;
rate_EU = -log(discount_EU)/ttm;

%%

spot_US = Market_US.spot;
spot_EU = Market_EU.spot;

drift_compensator_US = -1/kappa_US * (1 - sqrt( 1 - 2 * kappa_US * theta_US - kappa_US * sigma_US^2));
drift_compensator_EU = -1/kappa_EU * (1 - sqrt( 1 - 2 * kappa_EU * theta_EU - kappa_EU * sigma_EU^2));

g = randn(nSim, 3);

G_US = random('inverseGaussian', 1, ttm/nu_US,[nSim, 1]);
G_EU =  random('inverseGaussian', 1, ttm/nu_EU,[nSim, 1]);
G_Z = random('inverseGaussian', 1, ttm/nu_Z,[nSim, 1]);

Beta_US = theta_US - a_US * Beta_Z;
Beta_EU = theta_EU - a_EU * Beta_Z;

gamma_US = sqrt(sigma_US^2 - a_US^2*gamma_Z^2);
gamma_EU = sqrt(sigma_EU^2 - a_EU^2*gamma_Z^2);

drift_compensator_YEU = -1/nu_EU * (1 - sqrt( 1 - 2 * nu_EU * Beta_EU - nu_EU * gamma_EU^2));
drift_compensator_YUS = -1/nu_US * (1 - sqrt( 1 - 2 * nu_US * Beta_US - nu_US * gamma_US^2));
drift_compensator_Z   = -1/nu_Z  * (1 - sqrt( 1 - 2 * nu_Z * Beta_Z - nu_Z * gamma_Z^2));

Y_US = - gamma_US^2 * (+ Beta_US) .* G_US * ttm + gamma_US .* sqrt(ttm .* G_US) .* g(:,1);
Y_EU = - gamma_EU^2 * ( + Beta_EU) .* G_EU * ttm + gamma_EU .* sqrt(ttm .* G_EU) .* g(:,2);

Z = - gamma_Z^2 * ( + Beta_Z) .* G_Z * ttm + gamma_Z .* sqrt(ttm .* G_Z) .* g(:,3);

% Marginal processes
X_US = Y_US + a_US * Z;
X_EU = Y_EU + a_EU * Z;

S_EU = spot_EU * exp((rate_EU + drift_compensator_EU)*ttm + X_EU);
S_US = spot_US * exp((rate_US + drift_compensator_US)*ttm + X_US);

F0_EU = spot_EU/discount_EU;
F0_US = spot_US/discount_US;

% S_EU = F0_EU * exp(drift_compensator_EU*ttm + X_EU);
% S_US = F0_US * exp(drift_compensator_US*ttm + X_US);

ind_fun = (S_EU < 0.95 * spot_EU);

payoff = max(S_US - spot_US, 0) .* ind_fun;

price = discount_US * mean(payoff);

% % indicator function 
% Indicator_function = (spot_EU * exp((rate_EU + drift_compensator_EU)*ttm + X_EU)< 0.95 * spot_EU);

% payoff = max( spot_US * exp((rate_US + drift_compensator_US)*ttm + X_US) - spot_US, 0) .* Indicator_function;

% price = mean(discount_US * mean(payoff));

% print the price
disp('--------------------------------------------------------------------')
disp(['The price is: ', num2str(price)]);

end % end function levy pricing alternative