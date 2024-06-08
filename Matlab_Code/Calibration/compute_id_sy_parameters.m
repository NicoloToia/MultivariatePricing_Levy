function calibParams = compute_id_sy_parameters(sigma_US,kappa_US, theta_US,sigma_EU,...
                                                 kappa_EU, theta_EU, nu_Z, nu_US, nu_EU)
% This function computes the parameters of the idiosyncratic shock process and the systematic shock process of the model
%
% INPUTS
%
% sigma_US: volatility of the US market
% sigma_EU: volatility of the EU market
% kappa_US: volatility of the volatility of the US market
% kappa_EU: volatility of the volatility of the EU market
% theta_US: skewness of the volatility of the US market
% theta_EU: skewness of the volatility of the EU market
% nu_Z: volatility of volatility of the systematic shock process
% nu_US: volatility of volatility of the US market in the idiosyncratic shock process
% nu_EU: volatility of volatility of the EU market in the idiosyncratic shock process
%
% OUTPUTS
%
% calibParams: struct containing the calibrated parameters of the idiosyncratic shock process and the systematic shock process

% Define the optimization variables
id_sy_param = optimvar('id_sy_param', 4);

% Define the equations as constraints
equation_1 = id_sy_param(1) * id_sy_param(3) - (kappa_US * theta_US / nu_Z);
equation_2 = id_sy_param(2) * id_sy_param(3) - (kappa_EU * theta_EU / nu_Z);
equation_3 = kappa_US * sigma_US^2 - nu_Z * id_sy_param(1)^2 * id_sy_param(4)^2;
equation_4 = kappa_EU * sigma_EU^2 - nu_Z * id_sy_param(2)^2 * id_sy_param(4)^2;

% Define the optimization problem
prob = optimproblem;

% Add the equations as constraints to the problem
prob.Constraints.equation_1 = equation_1 == 0;
prob.Constraints.equation_2 = equation_2 == 0;
prob.Constraints.equation_3 = equation_3 == 0;
prob.Constraints.equation_4 = equation_4 == 0;

% Add the additional constraint id_sy_param(4) > 0
prob.Constraints.constraint_1 = id_sy_param(4) >= 0;

% Initial value
x0.id_sy_param = ones(4, 1);

% Options for the solver
options = optimoptions('fmincon', 'Display', 'off');

% Solve the optimization problem
solution = solve(prob, x0, 'Options', options);

% Extract the calibrated parameters
a_US = solution.id_sy_param(1);
a_EU = solution.id_sy_param(2);
Beta_Z = solution.id_sy_param(3);
gamma_Z = solution.id_sy_param(4);

% compute the other parameters using covolution equations
Beta_US = theta_US - a_US * Beta_Z;
Beta_EU = theta_EU - a_EU * Beta_Z;

gamma_US = sqrt(sigma_US^2 - a_US^2*gamma_Z^2);
gamma_EU = sqrt(sigma_EU^2 - a_EU^2*gamma_Z^2);

% Create a struct with all the calibrated parameters
calibParams.EU.a = a_EU;
calibParams.EU.Beta = Beta_EU;
calibParams.EU.gamma = gamma_EU;
calibParams.EU.nu = nu_EU;

calibParams.US.a = a_US;
calibParams.US.Beta = Beta_US;
calibParams.US.gamma = gamma_US;
calibParams.US.nu = nu_US;

calibParams.Z.Beta = Beta_Z;
calibParams.Z.gamma = gamma_Z;
calibParams.Z.nu = nu_Z;

end