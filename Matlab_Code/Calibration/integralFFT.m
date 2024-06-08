function I = integralFFT(phi, M, dz, queryPoints)
% This function computes the Fourier Transform of the input integrand
% using the FFT algorithm.
%
% INPUTS:
%   phi: The integrand (characteristic function)
%   M: The number of points in the grid
%   dz: The grid spacing
%   queryPoints: The points at which to interpolate the integral
%
% OUTPUTS:
%   I: The integral of the integrand

% compute N
N = 2^M;

% compute the z values
z_1 = -(N-1)/2 * dz;
z = z_1:dz:-z_1;

% compute the dxi value
d_xi = 2 * pi / (N * dz);
xi_1 = -(N-1)/2 * d_xi;
xi = xi_1:d_xi:-xi_1;

% use the lewis formula to compute the function to integrate
f = 1 / (2*pi) *  phi(-xi - 1i/2) ./ (xi.^2 + 1/4);
f_tilde = f .* exp(-1i * z_1 * d_xi .* (0:N-1));

% compute the FFT
FFT = fft(f_tilde);

% compute the prefactor
prefactor = d_xi * exp(-1i * xi_1 * z);

% compute the integral by multiplying by the prefactor
I = prefactor .* FFT;

%check that the immaginary part is close to zero
% %if not plot the value of immaginary part (10 ^  is a threshold)
% if max(abs(imag(I))) > 10^-3
%     % figure;
%     % plot(imag(I))
%     warning('Immaginary part of the integral is not close to zero')
%     % display the value of the immaginary part and the iteration number
%     disp(['Immaginary part of the integral is: ', num2str(max(abs(imag(I))))])
%     disp(['Iteration number: ', num2str(M)])  
% end

% get only the real part
I = real(I);

% interpolate the values
I = interp1(z, I, queryPoints);

end