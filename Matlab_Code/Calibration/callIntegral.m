function callPrices = callIntegral(B0, F0, sigma, kappa, eta, t, log_moneyness, M, dz, flag)
% Compute the price of a call option using the integral of Normal Mean-Variance Mixture model
%
% INPUT:
%   B0: discount factor at time 0
%   F0: forward price at time 0
%   sigma: variance of the model
%   kappa: vol of vol
%   eta: skewness
%   t: time to maturity
%   log_moneyness: log of the moneyness to compute the price at
%   M: N = 2^M, number of nodes for the FFT and quadrature
%   flag: flag to choose NIG or VG
%
% OUTPUT:
%   callPrices: price of the call option (same size as log_moneyness)

% Compute the compensator
compensator_NIG =  - t./kappa * (1-sqrt(1-2.*kappa.*eta - kappa.*sigma .^2));
compensator_VG  = t./kappa * log(1 - eta * kappa - (sigma^2 * kappa)/2);

% Select the model
if strcmp(flag, 'NIG')
    % charateristic function NIG
    phi = @(xi) exp(t.*(1/kappa * (1 - sqrt(1 - 2i .* xi .* kappa .* eta + xi.^2 .* kappa .* sigma.^2))))...
                .*exp(xi .* 1i .* compensator_NIG);
elseif strcmp(flag, 'VG')
    % characteristic function VG
    phi = @(xi) (1 - 1i .* xi .* eta .* kappa + (xi.^2 * sigma^2 * kappa / 2)).^(-t / kappa)...
                 .* exp(1i .* xi .*  compensator_VG);
else
    error('Flag not recognized');
end                

% compute the integral via fast fourier transform
I = integralFFT(phi, M, dz, log_moneyness);

% apply the lewis formula
callPrices = B0 * F0 * (1 - exp(-log_moneyness/2) .* I);
    
end