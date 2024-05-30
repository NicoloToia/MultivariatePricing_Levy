%% THE BIG LÉVY PROJECT: Multivariate Pricing
% Final Project Financial Engineering 2024
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
addpath('Pricing')

%% IMPORT DATA

% Set the settlement date
settlement = datenum("07/09/2023");

% Load the dataset
Markets = load('OptionData.mat');
Market_EU = Markets.mkt_EU;
Market_US = Markets.mkt;
clear Markets;
% Load the market returns
load('SPXSX5Ereturns.mat');

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

% Market_EU_filtered = compute_ImpVol(Market_EU_filtered, TTM_EU, rates_EU);

% Plot the filtered implied volatility smiles for the EU market
plot_ImpVol(Market_EU_filtered, 'EU OTM Implied Volatility Smile (Filtered)');
% Plot the filtered implied volatility smiles for the US market
plot_ImpVol(Market_US_filtered, 'US OTM Implied Volatility Smile (Filtered)');

close all;
%% CALIBRATION

% Define the weight of both markets (EU and US)
w_EU = spot_EU/(spot_EU + spot_US);
w_US = spot_US/(spot_EU + spot_US);

% Set the Fast Fourier Transform (FFT) parameters
M_fft = 15;
dz_fft = 0.005;
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
%p0 = [0.5 20 1 1 20 2];
p0 = [0.05 0.05 0.05 0.05 0.05 0.05];


% Non linear constraints
const = @(x) constraint(x, alpha);
% options
options = optimset('Display', 'iter');
% options = optimoptions('fmincon', 'Display', 'off');

% Optimization
%calibrated_param = fmincon(obj_fun, p0, A, b, [], [], [], [], const, options);
% M = 15, dz = 0.0005
% calibrated_param = [0.11908 0.0063921 0.018383 0.11747 0.009144 0.015161];
% M = 15, dz = 0.005
 calibrated_param = [0.11991 0.0024632 0.021422 0.10851 0.0019843 0.021599];

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
corrHist = corr(Returns.Annually(:,2), Returns.Annually(:,1));

% Define the system of equations
eqn1 = @(nu) calibrated_param(2) - (nu(1)*nu(3))/(nu(1) + nu(3));
eqn2 = @(nu) calibrated_param(5) - (nu(2)*nu(3))/(nu(2) + nu(3));

% Define the correlation function
rho = @(nu) corrHist - sqrt( (nu(1)*nu(2)) / ((nu(1) + nu(3)) * (nu(2) + nu(3))) );  

% Define the system of equations and solve it to find the calibrated parameter nu_z
system_eq = @(nu) [eqn1(nu), eqn2(nu), rho(nu)];
options = optimoptions('fsolve', 'Display', 'off');
nu_calibrated = fsolve(system_eq, ones(3,1), options);

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
[percentage_error_EU, percentage_error_US] = ...
        percentage_error(Market_EU_calibrated, Market_US_calibrated, Market_EU_filtered, Market_US_filtered);

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

% select the OTM implied volatilities for the EU market
Market_EU_calibrated = select_OTM(Market_EU_calibrated);

% select the OTM implied volatilities for the US market
Market_US_calibrated = select_OTM(Market_US_calibrated);

%% COMPUTE IMPLIED VOLATILITY PERCENTAGE ERRORS

% compute the percentage error (implied volatility) for the calibrated model
[percentage_error_EU_IV, percentage_error_US_IV] = ...
             percentage_error_ImpVol(Market_EU_calibrated, Market_US_calibrated, Market_EU_filtered, Market_US_filtered);

% print the results
disp(['The average percentage error for the EU market (Implied Volatility) is: ', num2str(percentage_error_EU_IV), '%']);
disp(['The average percentage error for the US market (Implied Volatility) is: ', num2str(percentage_error_US_IV), '%']);

%% PLOT IMPLIED VOLATILITIES FOR THE CALIBRATED PRICES

% Plot the model implied volatilities versus the market implied volatilities for the EU market
plot_model_ImpVol(Market_EU_calibrated, Market_EU_filtered, 'EU Market Model Implied Volatilities vs EU Market Implied Volatilities');

% Plot the model implied volatilities versus the market implied volatilities for the US market
%plot_model_ImpVol(Market_US_calibrated, Market_US_filtered, 'US Market Model Implied Volatilities vs US Market Implied Volatilities');

%%  ESTIMATE HISTORICAL CORRELATION BETWEEN THE TWO INDExES

% Plot the returns of the two markets yearly and daily
plot_returns(Market_EU, Market_US, Returns);

% Compute the historical correlation between the two markets with the yearly returns
HistCorr = corr(Returns.Annually(:,2), Returns.Annually(:,1));

% Print the results
disp('---------------------------------------------------------------------')
disp(['The historical correlation between the two indexes is: ', num2str(HistCorr)]);
disp('---------------------------------------------------------------------')

%% NEW STRUCT FOR MARKET MODEL (BLACK)

% EU Market
Market_EU_Black.datesExpiry = Market_EU_filtered.datesExpiry;
Market_EU_Black.strikes = Market_EU_filtered.strikes;
Market_EU_Black.spot = Market_EU_filtered.spot;
Market_EU_Black.F0 = Market_EU_filtered.F0;
Market_EU_Black.B_bar = Market_EU_filtered.B_bar;

% US Market
Market_US_Black.datesExpiry = Market_US_filtered.datesExpiry;
Market_US_Black.strikes = Market_US_filtered.strikes;
Market_US_Black.spot = Market_US_filtered.spot;
Market_US_Black.F0 = Market_US_filtered.F0;
Market_US_Black.B_bar = Market_US_filtered.B_bar;


%% Alternative Model: BLACK CALIBRATION

% Define the objective function for the black model
% EU market
EU_black_obj = @(sigma) black_obj(Market_EU_filtered, TTM_EU, sigma);

% US market
US_black_obj = @(sigma) black_obj(Market_US_filtered, TTM_US, sigma);

% options
options = optimoptions('fmincon', 'Display', 'off');

% Calibrate the Black model for the two markets
% Constraints: sigma > 0
% the initial guess is set to 0.001

% EU market
sigmaB_EU = fmincon(EU_black_obj, 0.0001, [], [], [], [], 0, [], [], options);

% US market
sigmaB_US = fmincon(US_black_obj, 0.0001, [], [], [], [], 0, [], [], options);

% print the results
disp('---------------------------------------------------------------------')
disp('The calibrated parameters are:');
disp(['sigmaB_EU = ', num2str(sigmaB_EU)]);
disp(['sigmaB_US = ', num2str(sigmaB_US)]);
disp('---------------------------------------------------------------------')

% add volatilities to the struct
Market_EU_Black.sigma = sigmaB_EU;
Market_US_Black.sigma = sigmaB_US;

% Compute the covariance of the Brownian motions and match the historical correlation between the two indexes
% covBMs = HistCorr * sigmaB_EU * sigmaB_US;
% %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MINCHIATA
% %!!!!!!!!!!!!!!!!!!!!!!!!!
% 
% % print the results
% disp('---------------------------------------------------------------------')
% disp(['The covariance between the BMs is: ', num2str(covBMs)]);
% disp('---------------------------------------------------------------------')

% compute the price of the derivative using the Black model

% EU market
% cycle through the expiries
for ii = 1:length(Market_EU_filtered.datesExpiry)
        % Compute the price of the derivative using the Black model use blk built in function and save in in a new struct
    [call , put] = blkprice(Market_EU_filtered.F0(ii).value, Market_EU_filtered.strikes(ii).value, rates_EU(ii), TTM_EU(ii), sigmaB_EU);
    Market_EU_Black.midCall(ii).value = call';
    Market_EU_Black.midPut(ii).value = put';
    
end

% US market
% cycle through the expiries
for ii = 1:length(Market_US_filtered.datesExpiry)
        % Compute the price of the derivative using the Black model use blk built in function and save in in a new struct
    [call , put] = blkprice(Market_US_filtered.F0(ii).value, Market_US_filtered.strikes(ii).value, rates_US(ii), TTM_US(ii), sigmaB_US);
    Market_US_Black.midCall(ii).value = call';
    Market_US_Black.midPut(ii).value = put';
end

% compute the percentage error for the Black model
[percentage_error_EU_Black, percentage_error_US_Black] = ...
             percentage_error(Market_EU_Black, Market_US_Black, Market_EU_filtered, Market_US_filtered);

% print the results
disp(['The average percentage error for the EU market (Black Model) is: ', num2str(percentage_error_EU_Black), '%']);
disp(['The average percentage error for the US market (Black Model) is: ', num2str(percentage_error_US_Black), '%']);

% Plot the model prices for the EU market versus real prices for each expiry

% Plot the model prices for the EU market versus real prices for each expiry
% plot_model_prices(Market_EU_Black, Market_EU_filtered, 'EU Market Model Prices vs EU Real Prices (Black Model)');

% Plot the model prices for the US market versus real prices for each expiry
% plot_model_prices(Market_US_Black, Market_US_filtered, 'US Market Model Prices vs US Real Prices (Black Model)');


%% PRICING USING BOTH MODELS: BLACK MODEL

% sigmaB_EU = 0.152;
% sigmaB_US = 0.1567;

% Compute the price of the derivative with the following payoff:
% Payoff = max(S1(t) - S1(0), 0)*I(S2(t) < 0.95*S2(0))
% where S1(t) and S2(t) are the prices of the two indexes at time t

% Set the target date, 1 year from the settlement date check for business days
targetDate = datetime(settlement, 'ConvertFrom', 'datenum') + calyears(1);
targetDate(~isbusday(targetDate, eurCalendar())) = busdate(targetDate(~isbusday(targetDate, eurCalendar())), 'modifiedfollow', eurCalendar());
targetDate = datenum(targetDate);

% Intialize the mean of the Brownian motions
MeanBMs = [0;
           0];

% Number of simulations
N_sim = 1e7;

% Compute the price of the derivative using the Black model
price_black = black_pricing(Market_EU_calibrated, spot_US, sigmaB_EU, sigmaB_US, settlement, targetDate, MeanBMs, HistCorr, N_sim);

% Compute the price of the derivative using the Lévy model
derivativePrice_MC = levy_pricing(Market_EU_calibrated, spot_US, settlement, targetDate, alpha,...
                                    kappa_EU, kappa_US, sigma_EU, sigma_US, theta_EU, theta_US, N_sim);
