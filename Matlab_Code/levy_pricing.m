function derivativePrice_MC = levy_pricing(Market_EU, S0_US, settlement, targetDate, alpha, kappa_EU, kappa_US, sigma_EU, sigma_US, theta_EU, theta_US, nSim)

    S0_EU = Market_EU.spot;
    % Simulation of the NIG processes
    % use a MonteCarlo simulation to compute the call prices
    Expiries = datenum([Market_EU.datesExpiry]');

    B_bar = [Market_EU.B_bar.value]';
    
    % Compute the discount
    discount = intExtDF(B_bar, Expiries, targetDate);
    
    % Compute the forward prices
    F0_EU = S0_EU/discount;
    F0_US = S0_US/discount;
    
    % Compute the time to maturity
    ACT_365 = 3;
    ttm = yearfrac(settlement, targetDate, ACT_365);

    % compute the Laplace exponent EU & US
    ln_L_EU = @(omega_EU) ttm/kappa_EU * (1 - alpha)/alpha * ...
        (1 - (1 + (omega_EU .* kappa_EU * sigma_EU^2)/(1-alpha)).^alpha );

    ln_L_US = @(omega_US) ttm/kappa_US * (1 - alpha)/alpha * ...
        (1 - (1 + (omega_US .* kappa_US * sigma_US^2)/(1-alpha)).^alpha );

    % draw the standard normal random variables
    g = randn(nSim, 2);
    % g = randn(nSim, 1);
    % draw the inverse gaussian random variables
    G = random('inversegaussian', 1, ttm/kappa_US, nSim, 1);
    
    % ft_EU = sqrt(ttm) * sigma_EU * sqrt(G) .* g(:,1) - (0.5 + theta_EU) * ttm * sigma_EU^2 * G - ln_L_EU(theta_EU);
    % ft_US = sqrt(ttm) * sigma_US * sqrt(G) .* g(:,2) - (0.5 + theta_US) * ttm * sigma_US^2 * G - ln_L_US(theta_US);

    ft_EU = sqrt(ttm) * sigma_EU * sqrt(G) .* g(:,1) - (0.5 + theta_EU) * ttm * sigma_EU^2 * G + ttm./kappa_EU * (1-sqrt(1-2.*kappa_EU.*theta_EU - kappa_EU.*sigma_EU .^2)) - ln_L_EU(theta_EU);
    ft_US = sqrt(ttm) * sigma_US * sqrt(G) .* g(:,2) - (0.5 + theta_US) * ttm * sigma_US^2 * G + ttm./kappa_US * (1-sqrt(1-2.*kappa_US.*theta_US - kappa_US.*sigma_US .^2)) - ln_L_US(theta_US);
    % compute F(t) EU & US
    % ft_EU = sqrt(ttm) * sigma_EU * sqrt(G) .* g(:,1) - (0.5 + theta_EU) * ttm * sigma_EU^2 * G;
    % ft_US = sqrt(ttm) * sigma_US * sqrt(G) .* g(:,2) - (0.5 + theta_US) * ttm * sigma_US^2 * G;
    
    % ft_EU = sqrt(ttm) * sigma_EU * sqrt(G) .* g - (0.5 + theta_EU) * ttm * sigma_EU^2 * G;
    % ft_US = sqrt(ttm) * sigma_US * sqrt(G) .* g - (0.5 + theta_US) * ttm * sigma_US^2 * G;
    
    S1_EU = F0_EU * exp(ft_EU); 

    S1_US = F0_US * exp(ft_US); 

    % indicator function EU & US
    
    Indicator_function = (S1_US < 0.95 * S0_US);

    % compute the derivative price
    prova = S1_EU - S0_EU;
    maggioridi0 = sum(prova>0);
    maggioridi02 = sum(Indicator_function);
    prova2 = max(S1_EU - S0_EU, 0);
    payoff = max(S1_EU - S0_EU, 0) .* Indicator_function;

    % Compute the price
    derivativePrice_MC = discount * mean(payoff);

%     % Confidence interval
% a = 0.01;
% CI = norminv(1-a)*std(payoff)/sqrt(nSim);
% priceCI = [price - CI, price + CI];

% Display the results
% fprintf('------------------------------------------------------------------\n');
% fprintf('The price of the derivative is: %.4f\n', derivativePrice_MC);
% fprintf('The confidence interval is: [%.4f, %.4f]\n', priceCI(1), priceCI(2));
% fprintf('------------------------------------------------------------------\n');

    
    % % numerical first 4 moments
    % mu_1 = mean(G);
    % mu_2 = mean(G.^2);
    % mu_3 = mean(G.^3);
    % mu_4 = mean(G.^4);
    % 
    % % analytical first 4 moments
    % mu = 1;
    % lambda = ttm/kappa;
    % mu_1_an = 1;
    % mu_2_an = mu^2 * (lambda + mu) / lambda;
    % mu_3_an = mu^3 * (lambda^2 + 3 * lambda * mu + 3 * mu^2) / lambda^2;
    % mu_4_an = 5 * mu^2 / lambda * mu_3_an + mu_2_an * mu^2;
    % 
    % print a table of the moments
    
    % disp('The first 4 moments of the inverse gaussian distribution are:');
    % disp(' ');
    % disp('Numerical | Analytical');
    % disp('-----------------------');
    % disp([mu_1, mu_1_an]);
    % disp([mu_2, mu_2_an]);
    % disp([mu_3, mu_3_an]);
    % disp([mu_4, mu_4_an]);


% Display the results
fprintf('------------------------------------------------------------------\n');
fprintf('The price of the derivative via Lévy is: %.4f\n', derivativePrice_MC);
% fprintf('The confidence interval is: [%.4f, %.4f]\n', priceCI(1), priceCI(2));
fprintf('------------------------------------------------------------------\n');

end
