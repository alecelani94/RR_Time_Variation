function draws = MCMC_TVPVAR(Data, P, priors, opts)
% MCMC_TVPVAR   Gibbs sampler for the UNRESTRICTED TVP-VAR(P) (no rank
% reduction). Used as a benchmark for MCMC_RR_TVPVAR.
%
% Model (non-centered, intercept folded in, full TV block):
%
%   y_t   = X_t * phi_bar + X_t * S * theta_t + eps_t,  eps_t ~ N(0, Sigma),
%   theta_t = theta_{t-1} + u_t,                          u_t   ~ N(0, I_M),
%
% with X_t = x_t' kron I_N,  x_t = [1, y_{t-1}', ..., y_{t-P}']',
% K = N*P + 1,  M = N*K,  S = diag(s)  (M x M).
%
% Sampler blocks (Gibbs):
%   1. phi_bar   -- system-wise Gaussian, Minnesota prior (shared w/ RR version)
%   2. lambda_own, lambda_cross -- hierarchical GIG (if priors.lam_hier)
%   3. theta     -- banded precision sampler (bandwidth M, total MT x MT)
%   4. s         -- Gaussian conjugate on the loading (M-dim)
%   5. Sigma     -- inverse-Wishart (full matrix)
%
% Memory note: the theta block requires a sparse MT x MT precision with
% bandwidth M. For large N (M ~ 1000) this becomes unwieldy and is the
% whole point of the rank-R approximation in MCMC_RR_TVPVAR.
%
% Inputs / outputs mirror MCMC_RR_TVPVAR (minus the R argument).

%% ---- setup ------------------------------------------------------------

[T_obs, N] = size(Data);  %#ok<ASGLU>
K = N * P + 1;                              % includes intercept column
M = N * K;                                   % total TV states per period

Y = Data(P+1:end, :);                       % T x N (LHS)
T = size(Y, 1);

% X: T x K  with row t = [1, y_{t-1}', y_{t-2}', ..., y_{t-P}']
X = ones(T, K);
for p = 1:P
    X(:, 1 + (p-1)*N + 1 : 1 + p*N) = Data(P+1-p : end-p, :);
end

K_lag = N * P;

%% ---- prior construction (built from user-facing hyperparameters) ------

[priors.scales, priors.Sigma_AR] = AR_scales(Y, 4);

[priors.V_phi0, priors.C_Psi] = construct_Minnesota( ...
    priors.scales, P, priors.lam(1), priors.lam(4));

own_mask = false(N, K_lag);
for l = 1:P
    for i = 1:N
        own_mask(i, (l-1)*N + i) = true;
    end
end
priors.ps_own   = find(own_mask(:));
priors.ps_cross = find(~own_mask(:));

priors.par_lam = [priors.sh_lam, priors.lam(2)/priors.sh_lam;     % own
                  priors.sh_lam, priors.lam(3)/priors.sh_lam];    % cross

priors.Sigma_nu = N + priors.Sigma_nu_offset;
priors.Sigma_S0 = (priors.Sigma_nu - N - 1) * priors.Sigma_AR;

%% ---- precomputes (fixed across the Gibbs loop) ------------------------

XX = X' * X;                                % K x K  (used in Block 1)

% First-difference operator on theta (block dim M, T blocks)
e1      = ones(T, 1);
H_T     = spdiags([-e1, e1], [-1, 0], T, T);    % T x T
H_theta = kron(H_T, speye(M));                  % MT x MT sparse
HtH     = H_theta' * H_theta;                    % MT x MT sparse  (RW prior precision)

%% ---- initialization ---------------------------------------------------

% Static coefficients: unrestricted OLS on the VAR(P) system.
Psi_bar      = (Y' * X) / (X' * X);          % N x K
E_ols        = Y - X * Psi_bar';
Sigma        = (E_ols' * E_ols) / (T - K);   % OLS residual covariance

% Hierarchical Minnesota lambdas: start at prior means.
lambda_own   = priors.lam(2);
lambda_cross = priors.lam(3);

% theta: simulate one RW path from the prior (innovation variance = 1).
theta        = cumsum(randn(M, T), 2);       % M x T

% Loading s = diag(S): draw from N(0, V_sl).
s            = sqrt(priors.V_sl) * randn(M, 1);

%% ---- storage ----------------------------------------------------------

iM           = M;                            % alias to avoid shadowing inside store
Mc           = opts.mcmc;
draws.Psi_bar      = zeros(Mc, N, K);
draws.lambda_own   = zeros(Mc, 1);
draws.lambda_cross = zeros(Mc, 1);
draws.theta        = zeros(Mc, iM, T);       % full TV block
draws.s            = zeros(Mc, iM);
draws.Sigma        = zeros(Mc, N, N);

%% ---- Gibbs loop -------------------------------------------------------

niter = opts.burn + Mc * opts.thin;
sIdx  = 0;
tic;
for iter = 1:niter

    %% Block 1: static coefficients Psi_bar ----------------------------
    % Residualize on TV component, then system-wise Gaussian draw.

    % S_l_m is the N x K matrix view of s (= reshape(s, N, K)).
    S_l_m = reshape(s, N, K);

    % TV contribution to y_t: (S_l_m .* Theta_t) * x_t, with Theta_t = reshape(theta(:,t), N, K)
    TV_part = zeros(T, N);
    for t = 1:T
        Theta_t      = reshape(theta(:, t), N, K);
        TV_part(t, :) = ((S_l_m .* Theta_t) * X(t, :)')';
    end
    Y_bar_psi = Y - TV_part;

    % Prior diagonal (NK x 1, column-stack of Psi_bar)
    V_lag_vec = priors.C_Psi(:);
    V_lag_vec(priors.ps_own)   = lambda_own   * V_lag_vec(priors.ps_own);
    V_lag_vec(priors.ps_cross) = lambda_cross * V_lag_vec(priors.ps_cross);
    V_psi_diag = [priors.V_phi0; V_lag_vec];

    Sigma_inv = Sigma \ eye(N);
    P_post    = spdiags(1 ./ V_psi_diag, 0, N*K, N*K) + kron(XX, Sigma_inv);
    rhs       = Sigma_inv * (Y_bar_psi' * X);
    rhs       = rhs(:);

    L_post   = chol(P_post, 'lower');
    mu_post  = L_post' \ (L_post \ rhs);
    psi_draw = mu_post + L_post' \ randn(N*K, 1);
    Psi_bar  = reshape(psi_draw, N, K);

    %% Block 2: lambda_own, lambda_cross (hierarchical Minnesota) ------
    if priors.lam_hier
        % TODO: GIG draws (port of Chan 2023, JBES line 147-150).
    end

    %% Block 3: theta (banded precision sampler) -----------------------
    % TODO: per-t Z_theta_t = X_t * diag(s), build sparse block-tridiagonal
    %       P_theta = HtH + blkdiag(Z_t' Sigma^{-1} Z_t), banded Cholesky draw.

    %% Block 4: s (Gaussian conjugate on the loading) ------------------
    % TODO: per-t W_t = X_t * diag(theta_t), build M x M precision
    %       P_s = V_sl^{-1} * I_M + sum_t W_t' Sigma^{-1} W_t, draw s.

    %% Block 5: Sigma (inverse-Wishart, full matrix) -------------------
    % Recompute TV contribution with the latest theta, s, then form residuals
    S_l_m = reshape(s, N, K);
    for t = 1:T
        Theta_t      = reshape(theta(:, t), N, K);
        TV_part(t, :) = ((S_l_m .* Theta_t) * X(t, :)')';
    end
    E_full = Y - X * Psi_bar' - TV_part;
    Sigma  = iwishrnd(priors.Sigma_S0 + E_full' * E_full, priors.Sigma_nu + T);

    %% store -----------------------------------------------------------
    if iter > opts.burn && mod(iter - opts.burn, opts.thin) == 0
        sIdx = sIdx + 1;
        draws.Psi_bar(sIdx, :, :)  = Psi_bar;
        draws.lambda_own(sIdx)     = lambda_own;
        draws.lambda_cross(sIdx)   = lambda_cross;
        draws.theta(sIdx, :, :)    = theta;
        draws.s(sIdx, :)           = s';
        draws.Sigma(sIdx, :, :)    = Sigma;
    end

    if mod(iter, opts.print_freq) == 0
        fprintf('Iter %d / %d  (elapsed %.1fs)\n', iter, niter, toc);
    end
end
fprintf('Total elapsed: %.2f sec\n', toc);

end
