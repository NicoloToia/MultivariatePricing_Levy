function Market = compute_prices(Market, TTM, M_fft, dz_fft, flag)
% This function computes the prices of the options using the calibrated
% parameters and the FFT method
%
% INPUTS
% Market: struct with the calibrated parameters
% TTM: time to maturity
% M_fft: 2^M_fft is the number of points for the FFT
% dz_fft: grid spacing for the FFT
% flag: NIG or VG
%
% OUTPUTS
% Market: struct with the prices of the options

% Call variables from the struct
sigma = Market.sigma;
kappa = Market.kappa;
theta = Market.theta;


% Cycle over the different expiries
for ii = 1:length(Market.datesExpiry)

    % Call variables from the struct
    F0 = Market.F0(ii).value;
    B_bar = Market.B_bar(ii).value;

    % log-Moneyness
    x = [log(F0 ./ Market.strikes(ii).value)]';

    % Compute the call prices with the FFT method
    Market.midCall(ii).value= callIntegral(B_bar, F0, sigma, kappa, theta, TTM(ii), x, M_fft, dz_fft, flag);

    % Use put call parity to compute the put prices
    Market.midPut(ii).value = Market.midCall(ii).value - B_bar * (F0 - Market.strikes(ii).value)';
    
end

end