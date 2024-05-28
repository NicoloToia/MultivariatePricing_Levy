%% THE BIG LEVY PROJECT: Multivariate Pricing
%  Final Project Financial Engineering 2024
% Professors: Roberto Baviera & Michele Azzone
% Group 2B
% Giacomo Manfredi  CP: 10776946
% Francesco Stillo  CP: 10698518
% Nicolò Toia       CP: 10628899
%

% clear the workspace
clear;
close all;
clc;

% fix the seed
rng(42); % the answer to everything

%% ADD PATHS

% add the path to the functions
addpath('Data');
addpath('Forward_Discounts');
addpath('ImpVol');
addpath('Filter');
addpath('Calibration');
addpath('Model_Check');

%% IMPORT DATA

% Set the settlement date
settlement = datenum("07/09/2023");

% Load the dataset
Markets = load('OptionData.mat');
Market_EU = Markets.mkt_EU;
Market_US = Markets.mkt;
clear Markets;
Returns = load('SPXSX5Ereturns.mat');

%% COMPUTE DISCOUNT FACTORS AND FORWARD PRICES FROM OPTION DATA

% Compute the market discount factors and forward prices
Market_EU = fwd_Bbar(Market_EU);
Market_US = fwd_Bbar(Market_US);

% Plot the forward prices (mid, Bid, Ask) for the EU market
%plot_fwd_prices(Market_EU);
% Plot the forward prices (mid, Bid, Ask) for the US market
%plot_fwd_prices(Market_US);

%% CREATE AUXILIARY VARIABLES AND COMPUTE MARKET ZERO RATES

% Import the spot price from the market data
spot_EU = Market_EU.spot;
spot_US = Market_US.spot;

% Import the forward prices from the market data
F0_EU = [Market_EU.F0.value]';
F0_US = [Market_US.F0.value]';

% Import the market discounts factors from the market data
discounts_EU = [Market_EU.B_bar.value]';
discounts_US = [Market_US.B_bar.value]';

% Year convenction ACT/365
ACT_365 = 3;

% Compute time to maturity (TTM) in year fractions
TTM_EU = yearfrac(settlement, Market_EU.datesExpiry, ACT_365);
TTM_US = yearfrac(settlement, Market_US.datesExpiry, ACT_365);

% Compute the market zero rates
rates_EU = -log(discounts_EU)./TTM_EU;
rates_US = -log(discounts_US)./TTM_US;

%% COMPUTE IMPLIED VOLATILITIES & SELECT OUT OF THE MONEY (OTM) OPTIONS

% Compute the implied volatilities for the EU market
Market_EU = compute_ImpVol(Market_EU, TTM_EU, rates_EU);

% Compute the implied volatilities for the US market
Market_US = compute_ImpVol(Market_US, TTM_US, rates_US);

% Select the OTM implied volatilities for the EU market
Market_EU = select_OTM(Market_EU);

% Select the OTM implied volatilities for the US market
Market_US = select_OTM(Market_US);

% Plot the implied volatility smiles for the EU market
plot_ImpVol(Market_EU, 'EU OTM Implied Volatility Smile');

% Plot the implied volatility smiles for the US market
plot_ImpVol(Market_US, 'US OTM Implied Volatility Smile');

%% FILTERING

% Compute the delta sensitivity for the EU market
Market_EU = sens_Delta(Market_EU, TTM_EU, rates_EU);
% Compute the delta sensitivity for the US market
Market_US = sens_Delta(Market_US, TTM_US, rates_US);

% Create a new struct for the EU market with the filtered options
Market_EU_filtered = Filter(Market_EU);
% Create a new struct for the US market with the filtered options
Market_US_filtered = Filter(Market_US);

% Plot the filtered implied volatility smiles for the EU market
plot_ImpVol(Market_EU_filtered, 'EU OTM Implied Volatility Smile (Filtered)');
% Plot the filtered implied volatility smiles for the US market
plot_ImpVol(Market_US_filtered, 'US OTM Implied Volatility Smile (Filtered)');

%% CALIBRATION

% Define the weight of both markets (EU and US)
w_EU = spot_EU/(spot_EU + spot_US);
w_US = spot_US/(spot_EU + spot_US);

% Set the Fast Fourier Transform (FFT) parameters
M_fft = 15;
dz_fft = 0.0005;
alpha = 0.5;

% Calibrate the NIG parameters for the two markets (EU and US)
% sigma_EU = p(1)
% kappa_EU = p(2)
% theta_EU = p(3)
% simga_US = p(4)
% kappa_US = p(5)
% theta_US = p(6)

% Compute Elapse Time
tic

% Define the objective function
obj_fun = @(p) objective_function(p, TTM_EU, TTM_US, w_EU, w_US, Market_EU_filtered, Market_US_filtered, M_fft, dz_fft, alpha);

% Linear constraints
A = [
    -1, 0, 0, 0, 0, 0;
    0, -1, 0, 0, 0, 0;
    0, 0, 0, -1, 0, 0;
    0, 0, 0, 0, -1, 0;
];

b = [
    0;
    0;
    0;
    0;
];

% Initial guess
%p0 = [ 0.37 11.8 0.09 0.36 32 0.04];
% p0 = [0.5 2 0.5 0.5 2 0.5];
p0 = [0.5 20 1 1 20 2];


% Non linear constraints
const = @(x) constraint(x, alpha);
% options
options = optimset('Display', 'iter');
% options = optimoptions('fmincon', 'Display', 'off');

% Optimization
calibrated_param = fmincon(obj_fun, p0, A, b, [], [], [], [], const, options);
%calibrated_param = [0.1679 0.18883 -0.018109 0.1295 0.34294 0.010364];

% End elapse time
toc

% print the results
disp('---------------------------------------------------------------------')
disp('The optimal parameters are:');
disp(['sigma_EU = ', num2str(calibrated_param(1))]);
disp(['kappa_EU = ', num2str(calibrated_param(2))]);
disp(['theta_EU = ', num2str(calibrated_param(3))]);
disp(['sigma_US = ', num2str(calibrated_param(4))]);
disp(['kappa_US = ', num2str(calibrated_param(5))]);
disp(['theta_US = ', num2str(calibrated_param(6))]);
disp('---------------------------------------------------------------------')
%%

% Set the Fast Fourier Transform (FFT) parameters
% M_fft = 15;
% dz_fft = 0.001;
% alpha = 0.5;
%calibrated_param = [0.1203 0.0002 0.0184 0.1199 0.0003 0.0144];
% 
% % nostri
% calibrated_param = [0.1679 0.18883 -0.018109 0.1295 0.34294 0.010364];

%% NEW STRUCT FOR MARKET MODEL

% Create a new struct for the EU market with the calibrated parameters
Market_EU_calibrated.sigma = calibrated_param(1);
Market_EU_calibrated.kappa = calibrated_param(2);
Market_EU_calibrated.theta = calibrated_param(3);

Market_EU_calibrated.datesExpiry = Market_EU_filtered.datesExpiry;
Market_EU_calibrated.strikes = Market_EU_filtered.strikes;
Market_EU_calibrated.spot = Market_EU_filtered.spot;
Market_EU_calibrated.F0 = Market_EU_filtered.F0;
Market_EU_calibrated.B_bar = Market_EU_filtered.B_bar;

% Create a new struct for the US market with the calibrated parameters
Market_US_calibrated.sigma = calibrated_param(4);
Market_US_calibrated.kappa = calibrated_param(5);
Market_US_calibrated.theta = calibrated_param(6);

Market_US_calibrated.datesExpiry = Market_US_filtered.datesExpiry;
Market_US_calibrated.strikes = Market_US_filtered.strikes;
Market_US_calibrated.spot = Market_US_filtered.spot;
Market_US_calibrated.F0 = Market_US_filtered.F0;
Market_US_calibrated.B_bar = Market_US_filtered.B_bar;


%% CALIBRATE THE SYSTEMATIC PARAMETER NU_Z

% nu1 = nu(1)
% nu2 = nu(2)
% nuZ = nu(3)

% Compute the historical correlation between the two markets
corrHist = corr(Returns.Returns.Annually(:,2), Returns.Returns.Annually(:,1));

% Define the system of equations
eqn1 = @(nu) calibrated_param(2) - (nu(1)*nu(3))/(nu(1) + nu(3));
eqn2 = @(nu) calibrated_param(5) - (nu(2)*nu(3))/(nu(2) + nu(3));

% Define the correlation function
rho = @(nu) corrHist - sqrt( (nu(1)*nu(2)) / ((nu(1) + nu(3)) * (nu(2) + nu(3))) );  

% Define the system of equations and solve it to find the calibrated parameter nu_z
system_eq = @(nu) [eqn1(nu), eqn2(nu), rho(nu)];
options = optimoptions('fsolve', 'Display', 'iter'); % Display iteration for debugging
nu_calibrated = fsolve(system_eq, [1, 1, 5], options);

% Compute the calibrated parameter nu_z using an alternative method
nu_z = sqrt(calibrated_param(2)*calibrated_param(5))/corrHist;

% disp the results
disp('---------------------------------------------------------------------')
disp('The calibrated parameters are:');
disp(['nu1 = ', num2str(nu_calibrated(1))]);
disp(['nu2 = ', num2str(nu_calibrated(2))]);
disp(['nuZ = ', num2str(nu_calibrated(3))]);

disp(['nuZ2 = ', num2str(nu_z)]);


%% COMPUTE PRICES VIA CALIBRATED PARAMETERS

% Choose the flag for the pricing method
flag = 'FFT';
%flag = 'quad';

% Rename the calibrated parameters for the EU market
sigma_EU = calibrated_param(1);
kappa_EU = calibrated_param(2);
theta_EU = calibrated_param(3);

% Rename the calibrated parameters for the US market
sigma_US = calibrated_param(4);
kappa_US = calibrated_param(5);
theta_US = calibrated_param(6);

% Compute the prices for EU market
Market_EU_calibrated = compute_prices(Market_EU_calibrated, TTM_EU, M_fft, dz_fft, alpha, flag);

% Compute the prices for US market
Market_US_calibrated = compute_prices(Market_US_calibrated, TTM_US, M_fft, dz_fft, alpha, flag);


%% CHECK FOR NEGATIVE PRICES

% Check for negative prices in the EU and US markets
check_neagtive_prices(Market_EU_calibrated, Market_US_calibrated);

%% COMPUTE PRICES PERCENTAGE ERRORS

% Compute the percentage errors for the EU and US markets
[percentage_error_EU, percentage_error_US] = percentage_error(Market_EU_calibrated, Market_US_calibrated, Market_EU_filtered, Market_US_filtered);

% Print the results
disp(['The average percentage error for the EU market is: ', num2str(percentage_error_EU), '%']);
disp(['The average percentage error for the US market is: ', num2str(percentage_error_US), '%']);


%% PLOT THE MODEL CALIBRATED PRICES VERSUS REAL PRICES FOR EACH EXPIRY

% Plot the model prices for the EU market versus real prices for each expiry
plot_model_prices(Market_EU_calibrated, Market_EU_filtered, 'EU Market Model Prices vs EU Real Prices');

% Plot the model prices for the US market versus real prices for each expiry
%plot_model_prices(Market_US_calibrated, Market_US_filtered, 'US Market Model Prices vs US Real Prices');


%% COMPUTE IMPLIED VOLATILITIES FOR THE CALIBRATED PRICES

% Compute the implied volatilities for the EU market
Market_EU_calibrated = compute_ImpVol(Market_EU_calibrated, TTM_EU, rates_EU);

% Compute the implied volatilities for the US market
Market_US_calibrated = compute_ImpVol(Market_US_calibrated, TTM_US, rates_US);

%% PLOT IMPLIED VOLATILITIES FOR THE CALIBRATED PRICES

% Plot the model implied volatilities versus the market implied volatilities for the EU market
plot_model_ImpVol(Market_EU_calibrated, Market_EU_filtered, 'EU Market Model Implied Volatilities vs EU Market Implied Volatilities');

% Plot the model implied volatilities versus the market implied volatilities for the US market
%plot_model_ImpVol(Market_US_calibrated, Market_US_filtered, 'US Market Model Implied Volatilities vs US Market Implied Volatilities');

