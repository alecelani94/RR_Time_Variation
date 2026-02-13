function results = MCMC_bilinear_TVP_VAR1(Y, P, lambda, Burn, MCMC, Thin)
% MCMC_bilinear_TVP_VAR1  Gibbs sampler for Bilinear TVP-VAR(1)
%
%   y_t = (Phi_bar + diag(a_t) * B * diag(c_t)) * y_{t-1} + eps_t
%   eps_t ~ N(0, Sigma),   B = b * ones(N)
%
% Inputs:
%   Y     - (T x N) data matrix
%   opts  - struct with fields:
%     .nburn      burn-in iterations      (default 2000)
%     .nsave      posterior draws to save  (default 5000)
%     .prior      struct with hyperparameters:
%       .lambda      Minnesota [const, own, cross, contemp, decay]
%       .kappa_b     prior variance for b             (default 1)
%       .sigma_a2    prior variance for a_{i,t}       (default 1)
%       .sigma_c2    prior variance for c_{j,t}       (default 1)
%       .nu0         IW degrees of freedom for Sigma  (default N+2)
%       .S0          IW scale matrix for Sigma        (default I_N)
%
% Outputs:
%   results - struct with:
%     .Phi_bar   (nsave x N^2)   posterior draws of vec(Phi_bar)
%     .b         (nsave x 1)     posterior draws of b
%     .Sigma     (nsave x N x N) posterior draws of Sigma
%     .a_mean    (T x N)         posterior mean of a_t
%     .c_mean    (T x N)         posterior mean of c_t

[T, N] = size(Y);
N2 = N^2;

% Minnesota prior for Phi_bar
scales     = AR_scales(Y, P);
[C_l, ~]   = construct_Minnesota(scales, P, lambda);
C_phi      = C_l(:, 2:N+1);               % N x N prior variances (no constant)

%% ---- Data -----------------------------------------------------------------
Xlag = Y(1:T-1, :);        % (T-1) x N   lagged y
Yobs = Y(2:T,   :);        % (T-1) x N   current y
Teff = T - 1;

%% ---- Initialise -----------------------------------------------------------
Phi_bar   = ((Xlag' * Xlag) \ (Xlag' * Yobs))';   % OLS
b         = 0.01;
a         = zeros(T, N);
c         = zeros(T, N);
resid     = Yobs - Xlag * Phi_bar';
Sigma     = (resid' * resid) / Teff;
Sigma_inv = Sigma \ eye(N);

%% ---- Storage --------------------------------------------------------------
Phi_bar_draws = zeros(nsave, N2);
b_draws       = zeros(nsave, 1);
Sigma_draws   = zeros(nsave, N, N);
a_post_mean   = zeros(T, N);
c_post_mean   = zeros(T, N);

IN = eye(N);

%% ---- Gibbs sampler --------------------------------------------------------
for iter = 1:ntot

    B_mat = b * ones(N);

    %% Block 1: Phi_bar | rest (Minnesota, equation by equation) -------------
    %  y_adj_t = y_t - diag(a_t)*B*diag(c_t)*y_{t-1}
    %  y_adj_t = Phi_bar * y_{t-1} + eps_t

    Y_adj = zeros(Teff, N);
    for t = 1:Teff
        tt = t + 1;
        Phi_tilde_t = diag(a(tt,:)) * B_mat * diag(c(tt,:));
        Y_adj(t,:)  = Yobs(t,:) - (Phi_tilde_t * Xlag(t,:)')';
    end

    for i = 1:N
        Q_i = diag(1 ./ C_phi(i,:));
        Phi_bar(i,:) = draw_mean_LRM(Y_adj(:,i), Xlag, [], [], Q_i)';
    end

    %% Block 2: b | rest -----------------------------------------------------
    %  e_t = b * w_t + eps_t,  w_t = diag(a_t)*ones(N)*diag(c_t)*y_{t-1}

    prec_b = 1 / kappa_b;
    info_b = 0;
    for t = 1:Teff
        tt  = t + 1;
        e_t = Yobs(t,:)' - Phi_bar * Xlag(t,:)';
        w_t = diag(a(tt,:)) * ones(N) * (c(tt,:)' .* Xlag(t,:)');
        prec_b = prec_b + w_t' * Sigma_inv * w_t;
        info_b = info_b + w_t' * Sigma_inv * e_t;
    end
    V_b = 1 / prec_b;
    b   = V_b * info_b + sqrt(V_b) * randn;
    B_mat = b * ones(N);

    %% Block 3: a_t | rest  (t = 2,...,T) ------------------------------------
    %  e_t = diag(h_t) * a_t + eps_t,  h_t = B*(c_t .* y_{t-1})

    prec_a0 = (1 / sigma_a2) * IN;
    for t = 1:Teff
        tt  = t + 1;
        e_t = Yobs(t,:)' - Phi_bar * Xlag(t,:)';
        h_t = B_mat * (c(tt,:)' .* Xlag(t,:)');
        H_t = diag(h_t);

        Prec_a = H_t * Sigma_inv * H_t + prec_a0;
        V_a    = Prec_a \ IN;
        mu_a   = V_a * (H_t * Sigma_inv * e_t);

        a(tt,:) = (mu_a + chol(V_a, 'lower') * randn(N, 1))';
    end

    %% Block 4: c_t | rest  (t = 2,...,T) ------------------------------------
    %  e_t = G_t * c_t + eps_t,  G_t(i,j) = a_i*B(i,j)*y_j(t-1)

    prec_c0 = (1 / sigma_c2) * IN;
    for t = 1:Teff
        tt  = t + 1;
        e_t = Yobs(t,:)' - Phi_bar * Xlag(t,:)';
        G_t = diag(a(tt,:)) * B_mat * diag(Xlag(t,:));

        Prec_c = G_t' * Sigma_inv * G_t + prec_c0;
        V_c    = Prec_c \ IN;
        mu_c   = V_c * (G_t' * Sigma_inv * e_t);

        c(tt,:) = (mu_c + chol(V_c, 'lower') * randn(N, 1))';
    end

    %% Block 5: Sigma | rest -------------------------------------------------
    E = zeros(Teff, N);
    for t = 1:Teff
        tt    = t + 1;
        Phi_t = Phi_bar + diag(a(tt,:)) * B_mat * diag(c(tt,:));
        E(t,:) = Yobs(t,:) - (Phi_t * Xlag(t,:)')';
    end

    S_post    = S0 + E' * E;
    nu_post   = nu0 + Teff;
    Sigma     = iwishrnd(S_post, nu_post);
    Sigma_inv = Sigma \ IN;

    %% Store ------------------------------------------------------------------
    if iter > nburn
        s = iter - nburn;
        Phi_bar_draws(s,:) = vec(Phi_bar)';
        b_draws(s)         = b;
        Sigma_draws(s,:,:) = Sigma;
        a_post_mean        = a_post_mean + a / nsave;
        c_post_mean        = c_post_mean + c / nsave;
    end

    if mod(iter, 1000) == 0
        fprintf('MCMC iteration %d / %d\n', iter, ntot);
    end
end

%% ---- Results --------------------------------------------------------------
results.Phi_bar = Phi_bar_draws;
results.b       = b_draws;
results.Sigma   = Sigma_draws;
results.a_mean  = a_post_mean;
results.c_mean  = c_post_mean;

end

%% ---- Local helper ----------------------------------------------------------
function val = get_opt(s, fld, default)
    if isfield(s, fld)
        val = s.(fld);
    else
        val = default;
    end
end
