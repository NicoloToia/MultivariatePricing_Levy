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
% nu_Z: volatility of the jumps of the common factor
%
% OUTPUTS
%
% a_US: coefficient of the systematic factor
% a_EU: coefficient of the systematic factor
% Beta_Z: parameter of the systematic shock process
% gamma_Z: parameter of the systematic shock process

% Define the optimization variables
id_sy_param = optimvar('id_sy_param',4);

% a_US = x(1);
% a_EU = x(2);
% Beta_Z = x(3);
% gamma_Z = x(4);

% Define the equations
equation_1 = id_sy_param(1) * id_sy_param(3) - (kappa_US * theta_US / nu_Z) == 0;
equation_2 = id_sy_param(2) * id_sy_param(3) - (kappa_EU * theta_EU / nu_Z) == 0;
equation_3 = kappa_US * sigma_US^2 - nu_Z * id_sy_param(1)^2 * id_sy_param(4) ^2  == 0;
equation_4 = kappa_EU * sigma_EU^2 - nu_Z * id_sy_param(2)^2 * id_sy_param(4) ^2  == 0;

% % Build a structure for the system of equations
% prob = eqnproblem;
% prob.Equations.eq1 = eq1;
% prob.Equations.eq2 = eq2;
% prob.Equations.eq3 = eq3;
% prob.Equations.eq4 = eq4;

% Build a system of equations
system = eqnproblem;
system.Equations.equation_1 = equation_1;
system.Equations.equation_2 = equation_2;
system.Equations.equation_3 = equation_3;
system.Equations.equation_4 = equation_4;

% Initial value
id_sy_param_0.id_sy_param = ones(4,1);

% Solve the system of equations
solution = solve(system,id_sy_param_0);

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