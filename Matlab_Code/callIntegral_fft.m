function callPrices = callIntegral_fft(B0, F0, alpha, sigma, kappa, eta, t, log_moneyness, M, dz)
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

% compute N
N = 2^M;

% compute the x values
z_1 = -(N-1)/2 * dz;
z = (z_1:dz:-z_1)';

% compute the dxi value
d_x = 2 * pi / (N * dz);
x_1 = -(N-1)/2 * d_x;
X = (x_1:d_x:-x_1)';

xi = - X - 1i/2;

% % Compute the characteristic function
% phi = @(xi) exp(t.*(1/kappa * (1 - sqrt(1 - 2i .* xi .* kappa .* eta + xi.^2 .* kappa .* sigma.^2))) ...
%     - t./kappa * (1-sqrt(1-2.*kappa.*eta - kappa.*sigma .^2))); 

phi = exp(t .* (1/kappa * (1 - sqrt(1 - 2i .* xi .* kappa .* eta + xi.^2 .* kappa .* sigma.^2))) ...
    - xi .* 1i .* t./kappa * (1-sqrt(1 - 2 .* kappa .* eta - kappa .* sigma.^2))); 

% use the lewis formula to compute the function to integrate
f = 1 / (2*pi)  *  phi ./ (X.^2 + 1/4);

j = (0:N-1)';
f_tilde = f .* exp(-1i * z_1 * d_x .* j);

% compute the FFT
FFT = fft(f_tilde);

% compute the prefactor
prefactor = d_x * exp(-1i * x_1 .* z);

% compute the integral by multiplying by the prefactor
I = prefactor .* FFT;

% check that the immaginary part is close to zero
% %if not plot the value of immaginary part (10 ^  is a threshold)

% if max(abs(imag(I))) > 10 ^ 10
%     figure;
%     plot(imag(I))
%     error('Immaginary part of the integral is not close to zero')   
% end

% get only the real part
I = real(I);

% interpolate the values
I = interp1(z, I, log_moneyness);

% apply the lewis formula
callPrices = B0 * F0 * (1 - exp(-log_moneyness/2) .* I);
    
end