function [c, ceq] = constraint(x, alpha)
% This is the constraint function for the "fmincon" Calibration. 
%
% INPUTS
% x: vector of parameters [sigma, kappa, theta]
% alpha: exponent of the model
%
% OUTPUTS
% c: inequality constraint
% ceq: equality constraint

% x(1) = sigma
% x(2) = kappa
% x(3) = theta

% Inequality constraint

% c= [- 0.5*x(2)*x(1)^2;
%     - 0.5*x(5)*x(4)^2 ];

% c= [- 0.5*x(2)*x(1)^2;
%     - 0.5 * x(5) * x(4)^2;
%     - (1-alpha) / (x(2)*x(1)^2) - x(3);
%     - (1-alpha) / (x(5)*x(4)^2) - x(6) ];

c = [];

% Equality constraint
ceq = x(1)^2/(x(3)^2*x(2)) - x(4)^2/(x(6)^2*x(5));

end