function price_closed_formula = closedFormula(Market_US, Market_EU, setDate, targetDate, rho)
% This function computes the price of a derivative with payoff:
% Payoff = max(S1(t) - S1(0), 0)*I(S2(t) < 0.95*S2(0))
% with a closed formula (the integral is numerically computed)
%
% INPUTS:
% Market_US: US market data (struct)
% Market_EU: EU market data (struct)
% setDate: settlement date
% targetDate: maturity of the derivative
% rho: correlation between the markets

ACT_365 = 3;

% import spot prices
S0_US = Market_US.spot;
S0_EU = Market_EU.spot;

% import expiries
Expiries_US = datenum([Market_US.datesExpiry]');
Expiries_EU = datenum([Market_EU.datesExpiry]');

% import discount factors
B_bar_US = [Market_US.B_bar.value]';
B_bar_EU = [Market_EU.B_bar.value]';

% import volatilities
sigma_US = Market_US.sigma;
sigma_EU = Market_EU.sigma;

% compute discounts
discount_US = intExtDF(B_bar_US, Expiries_US, targetDate);
discount_EU = intExtDF(B_bar_EU, Expiries_EU, targetDate);

% time to maturity
ttm = yearfrac(setDate, targetDate, ACT_365);

% zero rates for EU & US 
zeroRate_US = -log(discount_US)/ttm;
zeroRate_EU = -log(discount_EU)/ttm;

% boundary for the integral
x_min = -inf;
x_max = (log(0.95) - (zeroRate_EU - 0.5 * sigma_EU^2) * ttm) / sigma_EU;

A = @(omega) ((zeroRate_US-0.5*sigma_US.^2)*ttm + sigma_US*(ttm*(1-rho^2)*sigma_US + rho*omega))/(sigma_US*sqrt(ttm*(1-rho^2)));
B = @(omega) A(omega) - sqrt(ttm*(1-rho^2))*sigma_US;

% function to integrate
fun = @(omega) (exp(zeroRate_US*ttm - 0.5*sigma_US^2*rho^2.*ttm + sigma_US*rho*omega).*...
            cdf('Normal',A(omega), 0, sqrt(ttm)) - cdf('Normal', B(omega), 0, sqrt(ttm))).*...
            1/sqrt(2*pi*ttm).*exp(-0.5*omega.^2/ttm);

% integral
I = integral(fun, x_min, x_max);
% price closed formula
price_closed_formula = discount_US * S0_US *I;

end 