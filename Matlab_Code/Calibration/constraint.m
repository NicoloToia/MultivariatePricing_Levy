function [c, ceq] = constraint(x, alpha)
    % function that represents the constraint in the Least Squares Calibration
    %p1=sigma
    %p2 k, p3 theta
    
    %c= [- 0.5*x(2)*x(1)^2;
    %    - 0.5 * x(5) * x(4)^2;];
    c = [];

%        c= [- 0.5*x(2)*x(1)^2;
%         - 0.5 * x(5) * x(4)^2;
%         - (1-alpha) / (x(2)*x(1)^2) - x(3);
%         - (1-alpha) / (x(5)*x(4)^2) - x(6);];
    
    ceq = x(1)^2/(x(3)^2*x(2)) - x(4)^2/(x(6)^2*x(5));
end