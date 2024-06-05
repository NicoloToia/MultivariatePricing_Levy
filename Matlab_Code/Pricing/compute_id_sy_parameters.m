function [a_US , a_EU , Beta_Z , gamma_Z] = compute_id_sy_parameters(sigma_US,kappa_US, theta_US,sigma_EU, kappa_EU, theta_EU, nu_Z)
% This function computes the parameters of the idiosyncratic shock process and the systematic shock process in the model of the paper
%
% INPUTS
%
% sigma_US: the NIG volatility of the US Market
% 
%
% ------------------------------------------------------------




% Define the optimization variables
id_sy_param = optimvar('id_sy_param',4);

% a_US = x(1);
% a_EU = x(2);
% Beta_Z = x(3);
% gamma_Z = x(4);

% Define the equations
eq1 = id_sy_param(1) * id_sy_param(3) - (kappa_US * theta_US / nu_Z) == 0;
eq2 = id_sy_param(2) * id_sy_param(3) - (kappa_EU * theta_EU / nu_Z) == 0;
eq3 = kappa_US * sigma_US^2 - nu_Z * id_sy_param(1)^2 * id_sy_param(4) ^2  == 0;
eq4 = kappa_EU * sigma_EU^2 - nu_Z * id_sy_param(2)^2 * id_sy_param(4) ^2  == 0;

% Build a structure for the system of equations
prob = eqnproblem;
prob.Equations.eq1 = eq1;
prob.Equations.eq2 = eq2;
prob.Equations.eq3 = eq3;
prob.Equations.eq4 = eq4;

% Initial value
id_sy_param_0.id_sy_param = ones(4,1);

% Solve the system of equations
solution = solve(prob,id_sy_param_0);

% Extract the calibrated parameters
a_US = solution.id_sy_param(1);
a_EU = solution.id_sy_param(2);
Beta_Z = solution.id_sy_param(3);
gamma_Z = solution.id_sy_param(4);

end

