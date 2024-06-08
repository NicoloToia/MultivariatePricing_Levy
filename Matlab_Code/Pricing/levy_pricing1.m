function [price, priceCI] = levy_pricing1(Market_US, Market_EU, settlement, targetDate,...
                             calibrated_param, ID_SY_caliParm, N_sim, flag)
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
% N_sim: number of simulations
%
% OUTPUTS
% price: price of the option
% 
% param --> [a_US, a_EU, Beta_Z, gamma_Z]

% Idiosyncratic factors for the US and EU markets
a_US = ID_SY_caliParm.US.a;
a_EU = ID_SY_caliParm.EU.a;
Beta_US = ID_SY_caliParm.US.Beta;
Beta_EU = ID_SY_caliParm.EU.Beta;
gamma_US = ID_SY_caliParm.US.gamma;
gamma_EU = ID_SY_caliParm.EU.gamma;
nu_US = ID_SY_caliParm.US.nu;
nu_EU = ID_SY_caliParm.EU.nu;
% Sytematic factor
Beta_Z = ID_SY_caliParm.Z.Beta;
gamma_Z = ID_SY_caliParm.Z.gamma;
nu_Z = ID_SY_caliParm.Z.nu;

% Rename the calibrated parameters for the EU market
sigma_EU = calibrated_param(1);
kappa_EU = calibrated_param(2);
theta_EU = calibrated_param(3);

% Rename the calibrated parameters for the US market
sigma_US = calibrated_param(4);
kappa_US = calibrated_param(5);
theta_US = calibrated_param(6);

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

spot_US = Market_US.spot;
spot_EU = Market_EU.spot;

% Compute the drift compensators in the two model NIG and VG
if strcmp(flag, 'NIG')
    % NIG
    drift_compensator_US = -1/kappa_US * (1 - sqrt( 1 - 2 * kappa_US * theta_US - kappa_US * sigma_US^2));
    drift_compensator_EU = -1/kappa_EU * (1 - sqrt( 1 - 2 * kappa_EU * theta_EU - kappa_EU * sigma_EU^2));

    G_US = random('inverseGaussian', 1, ttm/nu_US,[N_sim, 1]);
    G_EU =  random('inverseGaussian', 1, ttm/nu_EU,[N_sim, 1]);
    G_Z = random('inverseGaussian', 1, ttm/nu_Z,[N_sim, 1]);

    % draw the standard normal random variables
    g = randn(N_sim, 3);

    % Idyosyncratic processes
    Y_US =   - gamma_US^2 * (0.5 + Beta_US) .* G_US * ttm + gamma_US .* sqrt(ttm .* G_US) .* g(:,1);
    Y_EU =  - gamma_EU^2 * (0.5 + Beta_EU) .* G_EU * ttm + gamma_EU .* sqrt(ttm .* G_EU) .* g(:,2);
    % Systematic process
    Z =  - gamma_Z^2 * ( 0.5 + Beta_Z) .* G_Z * ttm + gamma_Z .* sqrt(ttm .* G_Z) .* g(:,3);

elseif strcmp(flag, 'VG')
    % VG
    drift_compensator_US = 1/kappa_US * log(1 - theta_US * kappa_US - (sigma_US^2 * kappa_US)/2);
    drift_compensator_EU = 1/kappa_EU * log(1 - theta_EU * kappa_EU - (sigma_EU^2 * kappa_EU)/2);

    drift_compensator_YEU = 1/nu_EU * log(1 - nu_EU * Beta_EU - (nu_EU * gamma_EU^2)/2);
    drift_compensator_YUS = 1/nu_US * log(1 - nu_US * Beta_US - (nu_US * gamma_US^2)/2);
    drift_compensator_Z   = 1/nu_Z * log(1 - nu_Z * Beta_Z - (nu_Z * gamma_Z^2)/2);

    % draw the variance gamma random variables
    Y=zeros(N_sim,1);
    % Sample dS -> increments of a Gamma
    dS=nu_EU*icdf('gamma',rand(N_sim,1),ttm/nu_EU,1);
    % Sample the VG
    Y(:,1)=Y(:,1)+drift_compensator_YEU*ttm+(1)*dS+gamma_EU*sqrt(dS).*randn(N_sim,1);
    G_EU = Y;

    X=zeros(N_sim,1);
    % Sample dS -> increments of a Gamma
    dS=nu_US*icdf('gamma',rand(N_sim,1),ttm/nu_US,1);
    % Sample the VG
    X(:,1)=X(:,1)+drift_compensator_YUS*ttm+(1)*dS+gamma_US*sqrt(dS).*randn(N_sim,1);
    G_US = X;

    W=zeros(N_sim,1);
    % Sample dS -> increments of a Gamma
    dS=nu_US*icdf('gamma',rand(N_sim,1),ttm/nu_US,1);
    % Sample the VG
    W(:,1)=W(:,1)+drift_compensator_Z*ttm+(1)*dS+gamma_US*sqrt(dS).*randn(N_sim,1);
    G_Z = W;

        % draw the standard normal random variables
    g = randn(N_sim, 3);

    % Idyosyncratic processes
    Y_US =   real(- gamma_US^2 * (0.5 + Beta_US) .* G_US * ttm + gamma_US .* sqrt(ttm .* G_US) .* g(:,1));
    Y_EU =  real(- gamma_EU^2 * (0.5 + Beta_EU) .* G_EU * ttm + gamma_EU .* sqrt(ttm .* G_EU) .* g(:,2));
    % Systematic process
    Z =  real(- gamma_Z^2 * ( 0.5 + Beta_Z) .* G_Z * ttm + gamma_Z .* sqrt(ttm .* G_Z) .* g(:,3));
else
    error('Flag not recognized');
end

% Marginal processes
X_US = real(Y_US + a_US * Z);
X_EU = Y_EU + a_EU * Z;

% Compute the forward prices
F0_EU = spot_EU/discount_EU;
F0_US = spot_US/discount_US;

% Simulate the dynamics of the two markets
S_EU = F0_EU * exp(drift_compensator_EU*ttm + X_EU);
S_US = F0_US * exp(drift_compensator_US*ttm + X_US);

% Indicator function for EU market
ind_fun = (S_EU < 0.95 * spot_EU);

% Compute the payoff
payoff = max(S_US - spot_US, 0) .* ind_fun;

% Compute the price of the option
price = discount_US * mean(payoff);

% confidence interval
a = 0.01;
CI = norminv(1-a)*std(payoff)/sqrt(N_sim);
priceCI = [price - CI, price + CI];

end 