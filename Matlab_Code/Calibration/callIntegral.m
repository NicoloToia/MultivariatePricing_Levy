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

% Compute the Laplace exponent as a function of omega and alpha
% if alpha ~= 0 % consider Normal Inverse Gaussian (or others alpha different from zero)
%     ln_L = @(omega) t/kappa * (1 - alpha)/alpha * ...
%         (1 - (1 + (omega .* kappa * sigma^2)/(1-alpha)).^alpha );
% else % consider Variance Gamma
%     ln_L = @(omega) -t/kappa * log(1 + kappa * omega * sigma^2);
% end

% % compute the laPlace exponent at eta
% ln_L_eta = ln_L(eta);

% Compute the characteristic function
%phi = @(xi) exp(-1i * xi * ln_L_eta) .* exp( ln_L (0.5 * ((xi.^2) + 1i * (1+2*eta) .* xi)));
%phi = @(xi) exp((t/kappa) * (1 - sqrt(1 - 2*1i*xi*eta*kappa + xi.^2 * sigma^2 * kappa)));
phi = @(xi) exp(t.*(1/kappa * (1 - sqrt(1 - 2i .* xi .* kappa .* eta + xi.^2 .* kappa .* sigma.^2))) ...
    - t./kappa * (1-sqrt(1-2.*kappa.*eta - kappa.*sigma .^2))); 


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