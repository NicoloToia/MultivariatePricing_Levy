function price = levy_pricing_alternative(Market_US, Market_EU, settlement, targetDate, sigma_US, sigma_EU, kappa_US, kappa_EU,...
            theta_US, theta_EU, nu_US, nu_EU, nu_Z, nSim)
% function levy_pricing_alternative
%
% INPUTS:
% 
% 
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

% Compute the time to maturity
ACT_365 = 3;
ttm = yearfrac(settlement, targetDate, ACT_365);

% solve the non linear system
eqn1 = @(param) kappa_US * theta_US - nu_Z * param(1) * param(3);
eqn2 = @(param) kappa_EU * theta_EU - nu_Z * param(2) * param(3);
eqn3 = @(param) kappa_US * sigma_US^2 - nu_Z * param(1)^2 *param(4);
eqn4 = @(param) kappa_EU * sigma_EU^2 - nu_Z * param(2)^2 *param(4); 

% Define the system of equations and solve it to find the calibrated parameter
system_eq = @(param) [eqn1(param), eqn2(param), eqn3(param), eqn4(param)];
options = optimoptions('fsolve', 'Display', 'off');
param_calibrated = fsolve(system_eq, ones(4,1), options);

a_US = param_calibrated(1);
a_EU = param_calibrated(2);
Beta_Z = param_calibrated(3);
gamma_Z = param_calibrated(4);

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

gamma_US = sigma_US^2 - a_US^2*gamma_Z^2;
gamma_EU = sigma_EU^2 - a_EU^2*gamma_Z^2;

Y_US = Beta_US .* G_US * ttm + gamma_US .* sqrt(ttm .* G_US) .* g(:,1);
Y_EU = Beta_EU .* G_EU * ttm + gamma_EU .* sqrt(ttm .* G_EU) .* g(:,2);

Z = Beta_Z .* G_Z * ttm + gamma_Z .* sqrt(ttm .* G_Z) .* g(:,3);

% Marginal processes
X_US = Y_US + a_US * Z;
X_EU = Y_EU + a_EU * Z;


% indicator function 
Indicator_function = (spot_EU * exp((rate_EU + drift_compensator_EU)*ttm + X_EU)< 0.95 * spot_EU);

payoff = max( spot_US * exp((rate_US + drift_compensator_US)*ttm + X_US) - spot_US, 0) .* Indicator_function;

price = mean(discount_US * mean(payoff))

end % end function levy pricing alternative