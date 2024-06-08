function [c, ceq] = cosnt_Nu(nus, kappa_US, kappa_EU)
% This function is used to impose the constraints on the optimization
% problem. The constraints are the following:
% 1. kappa_EU - (nus(1)*nus(3))/(nus(1)+nus(3)) = 0
% 2. kappa_US - (nus(2)*nus(3))/(nus(2)+nus(3)) = 0
% 3. sqrt((nus(1) * nus(2)) / ((nus(1) + nus(3)) * (nus(2) + nus(3)))) - (sqrt(kappa_EU * kappa_US)/ nus(3)) = 0
%
% INPUTS:
% nus: vector of the parameters to be optimized
% kappa_EU: the kappa parameter for the EU market
% kappa_US: the kappa parameter for the US market
%
% OUTPUTS:
% c: inequality constraints
% ceq: equality constraints

% nu_US = nus(1);
% nu_EU = nus(2);
% nu_Z = nus(3);

% Inequality constraint
c = [];

% Equality constraints
ceq = [sqrt((nus(1) * nus(2)) / ((nus(1) + nus(3)) * (nus(2) + nus(3)))) - (sqrt(kappa_EU * kappa_US)/ nus(3));
        nus(1)*nus(3)/(nus(1)+nus(3)) - kappa_US;
        nus(2)*nus(3)/(nus(2)+nus(3)) - kappa_EU];

end