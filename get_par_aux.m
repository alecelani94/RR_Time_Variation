
% AR(1) coefficients for a(t) and c(t)
ar1_aux = [1;       % a(t): RW
           ar1_par]; % c(t): stationary AR(1)

% innovation variances for a(t) and c(t)
sigma2_aux = [T^(-1/2);                                         % RW:    Var_T = T * sigma2 = T^{1/2}
              T^(1/2) * (1 - ar1_par^2) / (1 - ar1_par^(2*T))]; % AR(1): Var_T = sigma2*(1-rho^{2T})/(1-rho^2) = T^{1/2}


par_aux = [ar1_aux sigma2_aux];

if pos_rw == 2
    par_aux = par_aux([2 1], :);
end
