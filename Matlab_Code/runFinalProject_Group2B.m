%% THE BIG LÉVY: Multivariate Pricing
%
% Final Project Financial Engineering 2024
% Professors: Roberto Baviera & Michele Azzone
%
% Group 2B
%
% Giacomo Manfredi  CP: 10776946    Mat: 247438
% Francesco Stillo  CP: 10698518    Mat: 247230
% Nicolò Toia       CP: 10628899    Mat: 247208

% clear the workspace
clear;
close all;
clc;
warning("off")

% fix the seed
rng(42); % The answer to life, the universe, and everything

% Compute Elapse Time
tic

%% ADD PATHS

% add the path to the functions
addpath('Data');
addpath('Forward_Discounts');
addpath('ImpVol');
addpath('Filter');
addpath('Calibration');
addpath('Model_Check');
addpath('Pricing')
addpath('Plot_3D')

%% IMPORT DATA

% Set the settlement date
settlement = datenum("07/09/2023");

% Load the dataset
Market_EU = load('OptionData.mat').mkt_EU;
Market_US = load('OptionData.mat').mkt;

% Load the market returns
load('SPXSX5Ereturns.mat');

%% MODEL SELECTION
% In this section, the user can select the model and the RMSE type to be used
% for the calibration of the model parameters.
% 
% The user can choose between the Normal Inverse Gaussian (NIG) and the
% Variance Gamma (VG) models.
%
% The second option is related to the computation of the Root Mean Square Error (RMSE):
%
%  ----> RMSE  : This is the simple version where the RMSE is computed without adjustment weights,
%                and the maturities of the US Market are capped at 19 (excluding the last maturity).
%
%  ----> RMSE2 : This is an updated version that allows calibration on all maturities. It considers the
%                error only when the model price is outside the bid-ask spread; otherwise, the
%                error is set to zero.

% Create a pop-up window to select the model
modelSelection = questdlg('Select the model:', 'Model Selection', 'NIG', 'VG', 'NIG');

% Check the user's choice
if isempty(modelSelection)
    disp('No model selected.');
end

% Create a pop-up window to select the RMSE type
rmseOptions = {'Unweighted', 'Filtered RMSE', 'TimeWindow'};
[selectedIndex, ok] = listdlg('PromptString', 'Select the model type:', ...
                              'SelectionMode', 'single', ...
                              'ListString', rmseOptions);

% Check the user's choice
if ok == 0
    disp('No RMSE type selected.');
else
    selectedRMSE = rmseOptions{selectedIndex};
end

% Display final choices
disp('---------------------------------------------------------------------');
fprintf('Final choice:\nModel: %s\nRMSE type: %s\n', modelSelection, selectedRMSE);
disp('---------------------------------------------------------------------');

% update the flag for the model selection
flag = modelSelection;

if strcmp(selectedRMSE, 'TimeWindow')
    flag_timeWindow = 1;
    flag_rmse = 'RMSE';
elseif strcmp(selectedRMSE, 'Unweighted')
    flag_timeWindow = 0;
    flag_rmse = 'RMSE';
else 
    flag_timeWindow = 0;
    flag_rmse = 'RMSE2';
end

%% COMPUTE DISCOUNT FACTORS AND FORWARD PRICES FROM OPTION DATA

% Compute the market discount factors and forward prices
Market_EU = fwd_Bbar(Market_EU);
Market_US = fwd_Bbar(Market_US);

% Plot the forward prices (mid, Bid, Ask) for the EU market
% plot_fwd_prices(Market_EU, 'EURO STOXX 50');
% Plot the forward prices (mid, Bid, Ask) for the US market
% plot_fwd_prices(Market_US, 'S&P 500');

%% OVERVIEW OF THE DATASET

% Compute the overview of the European market data
overview_EU = dataset_overview(Market_EU, 'EURO STOXX 50 Index Options');

% Compute the overview of the US market data
overview_US = dataset_overview(Market_US, 'S&P 500 Index Options');

%% CREATE AUXILIARY VARIABLES AND COMPUTE MARKET ZERO RATES

% Import the spot price from the market data
spot_EU = Market_EU.spot;
spot_US = Market_US.spot;

% Import the market discounts factors from the market data
discounts_EU = [Market_EU.B_bar.value]';
discounts_US = [Market_US.B_bar.value]';

% Import the forward prices from the market data
F0_EU = [Market_EU.F0.value]';
F0_US = [Market_US.F0.value]';

% Theorreical forward price, Garman-Kohlhagen formula
F0_EU_KG = spot_EU./discounts_EU;
F0_US_KG = spot_US./discounts_US;

% Plot the forward prices for the EU market
figure;
plot(Market_EU.datesExpiry, F0_EU, 'b', 'LineWidth', 1);
hold on;
plot(Market_EU.datesExpiry, F0_EU_KG, 'r--', 'LineWidth', 1);
ylabel('Forward Prices');
title('Forward Prices for the EURO STOXX 50');
legend('Forward Prices', 'G-K Forward Prices', 'Location', 'northwest');
grid on;
hold off;

% Plot the forward prices for the US market
figure;
plot(Market_US.datesExpiry, F0_US, 'b', 'LineWidth', 1);
hold on;
plot(Market_US.datesExpiry, F0_US_KG, 'r--', 'LineWidth', 1);  
ylabel('Forward Prices');
title('Forward Prices for the S&P 500');
legend('Forward Prices', 'G-K Forward Prices', 'Location', 'northwest');
grid on;
hold off;

% Year convenction ACT/365
ACT_365 = 3;

% Compute time to maturity (TTM) in year fractions
TTM_EU = yearfrac(settlement, Market_EU.datesExpiry, ACT_365);
TTM_US = yearfrac(settlement, Market_US.datesExpiry, ACT_365);

% Compute the market zero rates
rates_EU = -log(discounts_EU)./TTM_EU;
rates_US = -log(discounts_US)./TTM_US;

% Plot the zero rates for the EU market
figure;
yyaxis left;
plot(Market_EU.datesExpiry, rates_EU, 'b', 'LineWidth', 1);
hold on;
ylabel('Zero Rates');
yyaxis right;
plot(Market_EU.datesExpiry, discounts_EU, 'r', 'LineWidth', 1);
ylabel('Discount Factors');
title('Zero Rates and Discount Factors for the EURO STOXX 50');
legend('Zero Rates', 'Discount Factors', 'Location', 'Best');
% font size of the legend
set(findobj(gcf, 'type', 'legend'), 'fontsize', 12);
grid on;
hold off;

% Plot the zero rates for the US market
figure;
yyaxis left;
plot(Market_US.datesExpiry, rates_US, 'b', 'LineWidth', 1);
hold on;
ylabel('Zero Rates');
yyaxis right;
plot(Market_US.datesExpiry, discounts_US, 'r', 'LineWidth', 1);
ylabel('Discount Factors');
title('Zero Rates and Discount Factors for the S&P 500');
legend('Zero Rates', 'Discount Factors', 'Location', 'Best');
set(findobj(gcf, 'type', 'legend'), 'fontsize', 12);
grid on;
hold off;

%% COMPUTE IMPLIED VOLATILITIES & SELECT OUT OF THE MONEY (OTM) OPTIONS

% Compute the implied volatilities for the EU market
Market_EU = compute_ImpVol(Market_EU, TTM_EU, rates_EU);

% Compute the implied volatilities for the US market
Market_US = compute_ImpVol(Market_US, TTM_US, rates_US);

% Select the OTM implied volatilities for the EU market
Market_EU = select_OTM(Market_EU);

% Select the OTM implied volatilities for the US market
Market_US = select_OTM(Market_US);

%Plot the implied volatility smiles for the EU market
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

% Flatten the implied volatility smiles if needed
target_maturity_last = datenum(Market_US_filtered.datesExpiry(end));
Market_US_filtered = flatten_smile(Market_US_filtered, target_maturity_last, 1);

target_maturity_2 = datenum(Market_US_filtered.datesExpiry(end-2));
Market_US_filtered = flatten_smile(Market_US_filtered, target_maturity_2, 0);

% Plot the filtered implied volatility smiles for the EU market
plot_ImpVol(Market_EU_filtered, 'EU OTM Implied Volatility Smile (Filtered)');

% Plot the filtered implied volatility smiles for the US market
plot_ImpVol(Market_US_filtered, 'US OTM Implied Volatility Smile (Filtered)');

%% Volatility Surface Plot
% Run this section to plot the 3D implied volatility surface for the EU and US markets.

% Plot the 3D implied volatility surface for the EU market
% plot3d_impl_vol_singleSurf(Market_EU_filtered, 'Volatilities Surface (EURO STOXX 50)')

% Plot the 3D implied volatility surface for the US market
% plot3d_impl_vol_singleSurf(Market_US_filtered, 'Volatilities Surface (S&P 500)')

%% Pause Execution to Review the Plots
% The plots are displayed for user review. The user is prompted to review the plots,
% and then press any key in the command window to continue with the script execution.
% This step is performed before the calibration process to ensure the code execution remains manageable.

% Pause the execution to review the plots, then press any key to continue
msgbox('Please review the plots. When done, press any key in the command window to continue.', 'Review Plots', 'modal');
disp('---------------------------------------------------------------------');
disp('When you are ready to continue, press any key in the command window.');
pause;

% Close all open plots
close all;

% Continue with the rest of your code
disp('All plots have been closed. Continuing with the script execution...');
disp('---------------------------------------------------------------------');

%% CALIBRATION

% Define the weight of both markets (EU and US)
w_EU = spot_EU/(spot_EU + spot_US);
w_US = spot_US/(spot_EU + spot_US);

% Set the Fast Fourier Transform (FFT) hyperparameters
M_fft = 15;
dz_fft = 0.0025;

% Calibrate the NIG parameters for the two markets (EU and US)
% sigma_EU = p(1)
% kappa_EU = p(2)
% theta_EU = p(3)
% simga_US = p(4)
% kappa_US = p(5)
% theta_US = p(6)

% select the alpha according to the model
if strcmp(flag, 'NIG')
    alpha = 0.5;
elseif strcmp(flag, 'VG')
    alpha = 0;
else
    disp('Flag not found')
end

% Define the objective function
obj_fun = @(p) objective_function(p, TTM_EU, TTM_US, w_EU, w_US, Market_EU_filtered, Market_US_filtered,...
                                     M_fft, dz_fft, flag, flag_rmse, flag_timeWindow);

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
p0 = [0.1 0.1 -0.1 0.1 0.1 -0.1];%1
% p0 = 0.2*rand(1, 6);

% Non linear constraints    
const = @(x) constraint(x);

% lower bound & upper bound
lb = [0 0 -inf 0 0 -inf];
ub = [];

% options
options = optimoptions('fmincon',...
    'OptimalityTolerance', 1e-7, ...
    'TolFun', 1e-5, ...
    'ConstraintTolerance', 1e-5,...
    'Display', 'off');                  % set iter to look into calibration steps

% Display a message indicating the start of the calibration
fprintf('Calibration started...\n');

% Optimization
calibrated_param = fmincon(obj_fun, p0, A, b, [], [], lb, ub, const, options);

% Display a message indicating the end of the calibration
fprintf('Calibration completed.\n');

% Rename the calibrated parameters for the EU market
sigma_EU = calibrated_param(1);
kappa_EU = calibrated_param(2);
theta_EU = calibrated_param(3);

% Rename the calibrated parameters for the US market
sigma_US = calibrated_param(4);
kappa_US = calibrated_param(5);
theta_US = calibrated_param(6);

% Define the parameter names and corresponding values
parameter_names = {'Market', 'sigma', 'kappa', 'theta'};
parameter_values = {'EURO SSTOXX 50', sigma_EU, kappa_EU, theta_EU; 'S&P 500', sigma_US, kappa_US, theta_US};

% Create a table and print the calibrated parameters
T = table(parameter_values(:,1), parameter_values(:,2), parameter_values(:,3), parameter_values(:,4), ...
    'VariableNames', parameter_names);
disp(' ');
disp(' The calibrated parameters are:');
disp(T);
disp('---------------------------------------------------------------------');

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

% nu_US = nu(1);
% nu_EU = nu(2);
% nu_Z = nu(3);

% compute the historical correlation between the two indexes
corrHist = corr(Returns.Annually(:,2), Returns.Annually(:,1));

% define the objective function
obFun = @(nu) ( sqrt( nu(1)*nu(2) / ((nu(1) + nu(3)) * (nu(2) + nu(3)))) - corrHist )^2;

% define the  linear constraints
A = [-1 0 0; 
      0 -1 0; 
      0 0 -1]; 
b = [0; 0; 0];

% Set the bounds
lb = [0 0 max(kappa_US, kappa_EU)]; 
ub = [];

% define the non linear constraints
constNU = @(nu) cosnt_Nu(nu, kappa_US, kappa_EU);

% Set the options
options = optimoptions('fmincon', ...
    'Algorithm', 'sqp', ...                 % Optimization algorithm: 'sqp' (Sequential Quadratic Programming)
    'OptimalityTolerance', 1e-8, ...
    'ConstraintTolerance', 1e-6, ... 
    'StepTolerance', 1e-10, ...     
    'MaxIterations', 10000, ...  
    'MaxFunctionEvaluations', 3e4, ... 
    'Display', 'off');                  

% Initial guess
X0 = 0.5*ones(1,3);
% X0 = 0.2*ones(1,3);
% X0 = [4 0.1 4];

% calibration of the parameters
nu_calibrated = fmincon(obFun, X0, A, b, [], [], lb, ub, constNU, options);

% extract the calibrated parameters
nu_US = nu_calibrated(1);
nu_EU = nu_calibrated(2);
nu_Z = nu_calibrated(3);

% prnt the results
% disp('---------------------------------------------------------------------')
% disp(['nu_US = ', num2str(nu_US)]);
% disp(['nu_EU = ', num2str(nu_EU)]);
% disp(['nu_Z = ', num2str(nu_Z)]);
% disp('---------------------------------------------------------------------')

% print the results in a table
T = table(nu_US, nu_EU, nu_Z, 'VariableNames', {'nu_US', 'nu_EU', 'nu_Z'});
disp(' ');
disp(' The calibrated parameters are:');
disp(T);
disp('---------------------------------------------------------------------')

% rho = sqrt(kappa_EU*kappa_US)/nu_Z;
rho = sqrt(nu_US*nu_EU / ((nu_EU+nu_Z) * (nu_US+nu_Z)));
disp(['The calibrated correlation is: ', num2str(rho)]);
disp('---------------------------------------------------------------------');

% display rho improvements


%%
% % Define the historical correlation
% corrHist = corr(Returns.Annually(:,2), Returns.Annually(:,1));
% 
% % Define the objective function
% obFun = @(nu) ( sqrt( nu(1)*nu(2) / ((nu(1) + nu(3)) * (nu(2) + nu(3)))) - corrHist )^2;
% 
% % Define the constraints (if any)
% constNU = @(nu) cosnt_Nu(nu, kappa_US, kappa_EU, corrHist);
% 
% % Set the bounds
% lb = [0 0 max(kappa_US, kappa_EU)];
% ub = [];
% 
% % Set GA options
% options = optimoptions('ga', ...
%     'PopulationSize', 100, ...       % Size of the population
%     'MaxGenerations', 200, ...       % Maximum number of generations
%     'MaxStallGenerations', 50, ...   % Maximum number of stall generations
%     'FunctionTolerance', 1e-6, ...   % Function tolerance
%     'Display', 'iter', ...           % Display iteration information
%     'PlotFcn', @customGaplotbestf);  % Custom plot function
% 
% % Number of variables
% nvars = 3;
% 
% % Run the GA
% [nu_calibrated, fval] = ga(obFun, nvars, [], [], [], [], lb, ub, constNU, options);
% 
% % Extract the calibrated parameters
% nu_US = nu_calibrated(1);
% nu_EU = nu_calibrated(2);
% nu_Z = nu_calibrated(3);
% 
% % Print the results
% disp('---------------------------------------------------------------------')
% disp(['nu_US = ', num2str(nu_US)]);
% disp(['nu_EU = ', num2str(nu_EU)]);
% disp(['nu_Z = ', num2str(nu_Z)]);
% disp('---------------------------------------------------------------------')
% 
% % Calculate and display the correlation
% rho = sqrt(kappa_EU * kappa_US) / nu_Z;
% disp(rho)
% rho = sqrt(nu_US * nu_EU / ((nu_EU + nu_Z) * (nu_US + nu_Z)));
% disp(rho)

%% COMPUTE IDIOSYNCRATIC & SYSTEMIC PARAMETERS

% Compute the idiosyncratic and systemic parameters for the two markets
ID_SY_caliParm = compute_id_sy_parameters(sigma_US,kappa_US, theta_US,sigma_EU, kappa_EU, theta_EU, nu_Z, nu_US, nu_EU);

% extract the parameters
a_US = ID_SY_caliParm.US.a;
Beta_US = ID_SY_caliParm.US.Beta;
gamma_US = ID_SY_caliParm.US.gamma;
nu_US = ID_SY_caliParm.US.nu;

a_EU = ID_SY_caliParm.EU.a;
Beta_EU = ID_SY_caliParm.EU.Beta;
gamma_EU = ID_SY_caliParm.EU.gamma;
nu_EU = ID_SY_caliParm.EU.nu;

Beta_Z = ID_SY_caliParm.Z.Beta;
gamma_Z = ID_SY_caliParm.Z.gamma;
nu_Z = ID_SY_caliParm.Z.nu;

% Define the parameter names and corresponding values
parameter_names = {'Parameter', 'a', 'gamma', 'nu', 'Beta'};
parameter_values = {'Y_EU', a_EU, gamma_EU, nu_EU, Beta_EU; ...
                    'Y_US', a_US, gamma_US, nu_US, Beta_US; ...
                    'Z', '-', gamma_Z, nu_Z, Beta_Z};

% Create a table
T = table(parameter_values(:,1), parameter_values(:,2), parameter_values(:,3), parameter_values(:,4), parameter_values(:,5), ...
    'VariableNames', parameter_names);
disp(' ');
disp(' The calibrated parameters for the idiosyncratic and systemic factors are:');
disp(T);
disp('---------------------------------------------------------------------');

%% COMPUTE PRICES VIA CALIBRATED PARAMETERS

% Choose the flag for the pricing method

% Compute the prices for EU market
Market_EU_calibrated = compute_prices(Market_EU_calibrated, TTM_EU, M_fft, dz_fft, flag);

% Compute the prices for US market
Market_US_calibrated = compute_prices(Market_US_calibrated, TTM_US, M_fft, dz_fft, flag);

% Check for negative prices in the EU and US markets
check_neagtive_prices(Market_EU_calibrated, Market_US_calibrated);

%% COMPUTE PRICES PERCENTAGE ERRORS

% Compute the percentage errors for the EU and US markets
[percentage_error_EU, percentage_error_US] = ...
        percentage_error(Market_EU_calibrated, Market_US_calibrated, Market_EU_filtered, Market_US_filtered);

% Print the results
disp(' Lévy Model Calibration Errors Check:');
disp(' ');
disp(['The average percentage error for the EU market is: ', num2str(percentage_error_EU), '%']);
disp(['The average percentage error for the US market is: ', num2str(percentage_error_US), '%']);
disp('  ');

%% PLOT THE MODEL CALIBRATED PRICES VERSUS REAL PRICES FOR EACH EXPIRY

% Plot the model prices for the EU market versus real prices for each expiry
% plot_model_prices(Market_EU_calibrated, Market_EU_filtered, 'EU Model vs Market Prices');

% Plot the model prices for the US market versus real prices for each expiry
% plot_model_prices(Market_US_calibrated, Market_US_filtered, 'US Model vs Market Prices');

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
disp('---------------------------------------------------------------------');

%% PLOT IMPLIED VOLATILITIES FOR THE CALIBRATED PRICES

% Plot the model implied volatilities versus the market implied volatilities for the EU market
% plot_model_ImpVol(Market_EU_calibrated, Market_EU_filtered, 'EU Market Model Implied Volatilities vs EU Market Implied Volatilities');

% Plot the model implied volatilities versus the market implied volatilities for the US market
% plot_model_ImpVol(Market_US_calibrated, Market_US_filtered, 'US Market Model Implied Volatilities vs US Market Implied Volatilities');

%% Implied Volatility Surface Plot: Model vs Market

% EU Market
figure;
plot3D_impVol(Market_EU_filtered, Market_EU_calibrated, 'EU Market Implied Volatilities');
legend('MKT','MOD');
hold off;

% US Market
figure;
plot3D_impVol(Market_US_filtered, Market_US_calibrated, 'US Market Implied Volatilities');
legend('MKT','MOD');
hold off;

%%  ESTIMATE HISTORICAL CORRELATION BETWEEN THE TWO INDExES

% Plot the returns of the two markets yearly and daily
% plot_returns(Market_EU, Market_US, Returns);

% Compute the historical correlation between the two markets with the yearly returns
HistCorr = corr(Returns.Annually(:,2), Returns.Annually(:,1));

% Print the results
disp(' ');
disp(['The historical correlation between the EURO STOXX 50 and the S&P 500 is: ', num2str(HistCorr)]);
disp(' ');
disp('---------------------------------------------------------------------');

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

% print the results in a table
T = table(sigmaB_EU, sigmaB_US, 'VariableNames', {'sigmaB_EU', 'sigmaB_US'});
disp(' ');
disp(' The calibrated parameters for the Black model are:');
disp(T);

% add volatilities to the struct
Market_EU_Black.sigma = sigmaB_EU;
Market_US_Black.sigma = sigmaB_US;

% compute the prices for the two markets using the Black model to compare with the real prices
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
disp(' ');
disp('Black Model Calibration Errors Check:')
disp(' ');
disp(['The average percentage error for the EU market (Black Model) is: ', num2str(percentage_error_EU_Black), '%']);
disp(['The average percentage error for the US market (Black Model) is: ', num2str(percentage_error_US_Black), '%']);
disp('---------------------------------------------------------------------');

% Plot the model prices for the EU market versus real prices for each expiry

% Plot the model prices for the EU market versus real prices for each expiry
% plot_model_prices(Market_EU_Black, Market_EU_filtered, 'EU Market Model Prices vs EU Real Prices (Black Model)');

% Plot the model prices for the US market versus real prices for each expiry
% plot_model_prices(Market_US_Black, Market_US_filtered, 'US Market Model Prices vs US Real Prices (Black Model)');

%% PRICING OF THE DERIVATIVE

% Compute the price of the derivative with the following payoff:
% Payoff = max(S1(t) - S1(0), 0)*I(S2(t) < 0.95*S2(0))
% where S1(t) and S2(t) are the prices of the two indexes at time t

% Set the target date, 1 year from the settlement date check for business days
targetDate = datetime(settlement, 'ConvertFrom', 'datenum') + calyears(1);
targetDate(~isbusday(targetDate, eurCalendar())) = busdate(targetDate(~isbusday(targetDate, eurCalendar())), 'modifiedfollow', eurCalendar());
targetDate = datenum(targetDate);

% Number of simulations
N_sim = 1e7;

%% PRICING VIA THE BLACK MODEL

% Intialize the mean of the Brownian motions
MeanBMs = [0;
           0];

% Compute the price of the derivative using the Black model
[price_black, CI_black] = black_pricing(Market_US_Black, Market_EU_Black, settlement, targetDate, MeanBMs, HistCorr, N_sim);

%% PRICING VIA THE LÉVY MODEL

% Compute the price of the derivative using the Lévy model
[price_levy1, CI_levy1] = levy_pricing1(Market_US_calibrated, Market_EU_calibrated, settlement, targetDate,...
                                     calibrated_param, ID_SY_caliParm, N_sim, flag);

%% PRICING VIA THE LÉVY MODEL (ALTERNATIVE)

% Compute the correlation between the two indexes, given the calibrated parameters
% rho = sqrt(kappa_EU*kappa_US)/nu_Z;
rho = sqrt(nu_US*nu_EU / ((nu_EU+nu_Z) * (nu_US+nu_Z)));

% Compute the price of the derivative using the Lévy model
[price_levy2, CI_levy2] = levy_pricing2(Market_US_calibrated, Market_EU_calibrated, settlement, targetDate, ...
                                    kappa_US, kappa_EU, sigma_US, sigma_EU, theta_US, theta_EU, rho, N_sim, flag);

%% PRICING VIA THE SEMI-CLOSED FORMULA

% price via the semi closed formula
price_closed_formula = closedFormula(Market_US_Black, Market_EU_Black, settlement, targetDate, HistCorr);

%% PRINT THE DERIVATE PRICES AND CONFIDENCE INTERVALS
% Define the parameter names and corresponding values
method_names = {'Black Model', 'Lévy Model', 'Alternative Lévy', 'Semi-Closed Formula'};

% Define the prices
prices = [price_black, price_levy1, price_levy2, price_closed_formula];

% Define the confidence intervals (NaN for those without a confidence interval)
CI = {CI_black, CI_levy1, CI_levy2, NaN(1, 2)};

% Create a cell array to store the results
results = cell(length(method_names), 3);

for i = 1:length(method_names)
    results{i, 1} = method_names{i};
    results{i, 2} = prices(i);
    if isnan(CI{i}(1))
        results{i, 3} = 'N/A';
    else
        results{i, 3} = sprintf('[%.4f, %.4f]', CI{i}(1), CI{i}(2));
    end
end

% Print the results in a table format
disp(' ');
disp('The prices of the derivative and their confidence intervals are:')
disp(' ');
disp('Method               | Price       | Confidence Interval');
disp('---------------------------------------------------------');
for i = 1:length(method_names)
    fprintf('%-20s | %.4f     | %s\n', results{i, 1}, results{i, 2}, results{i, 3});
end

disp(' ');

% End elapsed time and store the result
elapsedTime = toc;

% Calculate minutes and seconds
minutes = floor(elapsedTime / 60);
seconds = mod(elapsedTime, 60);

% Display the elapsed time in a nice format
disp('---------------------------------------------------------------------');
fprintf('Elapsed time: %d minutes and %.2f seconds\n', minutes, seconds);
disp('---------------------------------------------------------------------');