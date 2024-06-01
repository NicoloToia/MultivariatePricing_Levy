function callPrices = callIntegral(B0, F0, alpha, sigma, kappa, eta, t, log_moneyness, M, dz, flag)
% Compute the price of a call option using the integral of Normal Mean-Variance Mixture model
%
% INPUT:
%   B0: discount factor at time 0
%   F0: forward price at time 0
%   alpha: exponent of the model
%   sigma: variance of the model
%   kappa: vol of vol
%   eta: skewness
%   t: time to maturity
%   log_moneyness: log of the moneyness to compute the price at
%   M: N = 2^M, number of nodes for the FFT and quadrature
%   flag: flag to choose the integration method ("FFT" or "quad")
%
% OUTPUT:
%   callPrices: price of the call option (same size as log_moneyness)

% % Compute the characteristic function
% phi = @(xi) exp(t.*(1/kappa * (1 - sqrt(1 - 2i .* xi .* kappa .* eta + xi.^2 .* kappa .* sigma.^2))) ...
%     - t./kappa * (1-sqrt(1-2.*kappa.*eta - kappa.*sigma .^2))); 

phi = @(xi) exp(t.*(1/kappa * (1 - sqrt(1 - 2i .* xi .* kappa .* eta + xi.^2 .* kappa .* sigma.^2))) ...
    - xi .* 1i .* t./kappa * (1-sqrt(1-2.*kappa.*eta - kappa.*sigma .^2))); 

% Compute the integral with the flag
if strcmp(flag, 'FFT')
    I = integralFFT(phi, M, dz, log_moneyness);
    
elseif strcmp(flag, 'quad')
    I = integralQuad(phi, M, dz, log_moneyness);
else
    error('Flag not recognized');
end

% apply the lewis formula
callPrices = B0 * F0 * (1 - exp(-log_moneyness/2) .* I);
    
end