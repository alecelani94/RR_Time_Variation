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
% Sampler blocks (Gibbs, Chan 2023 JBES joint update + FS-Wagner sign flip):
%   1. (phi_bar, s)              -- JOINT Gaussian conjugate (Chan 2023);
%                                    replaces the separate static + loading
%                                    blocks of a vanilla NC sampler.
%   2. lambda_own, lambda_cross  -- hierarchical GIG (if priors.lam_hier)
%   3. theta                     -- banded precision sampler (bandwidth M)
%   3.5 sign flip                -- jointly flip (s_j, theta_{j,:}) w.p. 1/2
%                                    (Fruhwirth-Schnatter & Wagner 2010)
%   4. Sigma                     -- inverse-Wishart (full matrix)
%
% Memory note: the theta block requires a sparse MT x MT precision with
% bandwidth M. For large N (M ~ 1000) this becomes unwieldy and is the
% whole point of the rank-R approximation in MCMC_RR_TVPVAR.

%% ---- setup ------------------------------------------------------------

[T_obs, N] = size(Data);  %#ok<ASGLU>
K = N * P + 1;                              % includes intercept column
M = N * K;                                  % total TV states per period

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

% Per-element prior variance of the diagonal loading s (M x 1, column-stack of
% N x K). First N entries are intercept loadings, remaining N*K_lag are lag.
V_sl_vec = [priors.V_sl_intercept * ones(N, 1); ...
            priors.V_sl_lag       * ones(N * K_lag, 1)];

%% ---- precomputes (fixed across the Gibbs loop) ------------------------

% Time-ordered y stack (used in joint block and theta block)
y_full = reshape(Y', N*T, 1);

% X broadcast shape for sparse-value building (1 x K x T)
X_broad = reshape(X', 1, K, T);

% X reshape for pagemtimes in Block 4 (K x 1 x T column-vectors)
X_cols  = reshape(X', K, 1, T); %#ok<NASGU> read inside Gibbs loop

% Static block of the joint design has CONSTANT entries X(t,k) (no theta);
% precompute its sparse values once.
vv_Zs = reshape(repmat(X_broad, N, 1, 1), [], 1);    % N*K*T x 1

% First-difference operator on theta blocks (fixed across iterations)
e1     = ones(T, 1);
H_T    = spdiags([-e1, e1], [-1, 0], T, T);
H_th   = kron(H_T, speye(M));
HtH_th = H_th' * H_th;

% Sparse-triplet (row, col) indices for Z_theta and the (Z_s, W) joint design.
% Order: (i, k, t) with i innermost -- matches reshape(*, N, K, T)(:) ordering.
idx_z   = (0:(N*K*T - 1))';
i_z     = mod(idx_z, N);
k_z     = mod(floor(idx_z / N), K);
t_z     = floor(idx_z / (N*K));
ii_Z_pre = t_z * N + i_z + 1;                 % row in NT-stack
jj_Z_pre = t_z * M + k_z * N + i_z + 1;       % col in MT-stack (for theta block)
jj_W_pre =           k_z * N + i_z + 1;       % col in M-stack  (per-coef pos)
clear idx_z i_z k_z t_z

% Doubled row/col indices for the joint design [Z_s, W] of size NT x 2M
ii_joint = [ii_Z_pre; ii_Z_pre];
jj_joint = [jj_W_pre; jj_W_pre + M];

% GIG hyperparameters for the hierarchical Minnesota update (fixed across iters)
p_own_gig     = priors.par_lam(1,1) - numel(priors.ps_own)   / 2;
p_cross_gig   = priors.par_lam(2,1) - numel(priors.ps_cross) / 2;
psi_own_gig   = 2 / priors.par_lam(1,2);
psi_cross_gig = 2 / priors.par_lam(2,2);

%% ---- initialization ---------------------------------------------------

% Static coefficients: unrestricted OLS on the VAR(P) system.
Psi_bar      = (Y' * X) / (X' * X);          % N x K
E_ols        = Y - X * Psi_bar';
Sigma        = (E_ols' * E_ols) / (T - K);   % OLS residual covariance

% Hierarchical Minnesota lambdas: start at prior means.
lambda_own   = priors.lam(2);
lambda_cross = priors.lam(3);

% theta: simulate one RW path from the prior (innovation variance = 1).
% Used by the very first Block 1 draw to seed the design matrix.
theta        = cumsum(randn(M, T), 2);       % M x T

% s is sampled jointly with Psi_bar in Block 1; no init needed.

%% ---- storage ----------------------------------------------------------

iM           = M;                            % alias to avoid shadowing inside store
Mc           = opts.mcmc;
draws.Psi_bar = zeros(Mc, N, K);
draws.lambda  = zeros(Mc, 2);                % [own, cross]
draws.theta   = zeros(Mc, iM, T);
draws.s       = zeros(Mc, iM);
draws.Sigma   = zeros(Mc, N, N);

%% ---- Gibbs loop -------------------------------------------------------

niter = opts.burn + Mc * opts.thin;
sIdx  = 0;
tic;
for iter = 1:niter

    %% Block 1 (JOINT): static coefficients phi_bar AND loading s -------
    % Both enter linearly:
    %   y_t = X_t * phi_bar + X_t * diag(theta_t) * s + eps_t
    %       = [X_t,  X_t * diag(theta_t)] * [phi_bar; s] + eps_t
    %       = Z_tilde_t * mu + eps_t,    mu := (phi_bar; s) in R^{2M}.
    %
    % Joint conditional posterior:
    %   mu | . ~ N(P^{-1} Z_tilde' (I_T kron Sigma^-1) y,  P^{-1})
    %   P  = blkdiag(V_phi^-1, V_s^-1)  +  Z_tilde' (I_T kron Sigma^-1) Z_tilde.
    %
    % This block absorbs the FS-Wagner sign indeterminacy (no separate
    % sign-flip step is needed) and captures the level-loading posterior
    % correlation that a sequential split would miss.

    % --- Build sparse joint design Z_tilde = [Z_s, W], NT x 2M ---
    theta_NKT = reshape(theta, N, K, T);
    vv_W      = reshape(theta_NKT .* X_broad, [], 1);     % N*K*T x 1
    Z_tilde   = sparse(ii_joint, jj_joint, ...
                       [vv_Zs; vv_W], N*T, 2*M);

    % --- Joint prior diagonal (2M x 1) ---
    V_lag_vec = priors.C_Psi(:);
    V_lag_vec(priors.ps_own)   = lambda_own   * V_lag_vec(priors.ps_own);
    V_lag_vec(priors.ps_cross) = lambda_cross * V_lag_vec(priors.ps_cross);
    V_psi_diag = [priors.V_phi0; V_lag_vec];
    V_mu_diag  = [V_psi_diag; V_sl_vec];

    % --- Sigma^-1 and block-diag over t ---
    Sigma_inv   = Sigma \ eye(N);
    Sigma_inv_T = kron(speye(T), Sigma_inv);

    % --- Posterior precision (dense 2M x 2M) and right-hand side ---
    P_post = spdiags(1 ./ V_mu_diag, 0, 2*M, 2*M) ...
           + Z_tilde' * Sigma_inv_T * Z_tilde;
    rhs    = Z_tilde' * (Sigma_inv_T * y_full);

    mu      = draw_precision(rhs, P_post);
    Psi_bar = reshape(mu(1:M),   N, K);
    s       = mu(M+1:end);

    %% Block 2: lambda_own, lambda_cross (hierarchical Minnesota) ------
    if priors.lam_hier
        Psi_lag      = Psi_bar(:, 2:end);
        sq_std_full  = Psi_lag(:).^2 ./ priors.C_Psi(:);
        lambda_own   = gigrnd(p_own_gig,   psi_own_gig, ...
                              sum(sq_std_full(priors.ps_own)),   1);
        lambda_cross = gigrnd(p_cross_gig, psi_cross_gig, ...
                              sum(sq_std_full(priors.ps_cross)), 1);
    end

    %% Block 3: theta (banded precision sampler) -----------------------
    % Z_theta_t = X_t * diag(s); see paper App. eq. for the banded structure.
    s_NK    = reshape(s, N, K);
    vv_Z    = reshape(s_NK .* X_broad, [], 1);
    Z_theta = sparse(ii_Z_pre, jj_Z_pre, vv_Z, N*T, M*T);

    y_bar_mat = Y - X * Psi_bar';
    y_bar_vec = reshape(y_bar_mat', N*T, 1);

    P_theta = HtH_th + Z_theta' * Sigma_inv_T * Z_theta;
    rhs_th  = Z_theta' * (Sigma_inv_T * y_bar_vec);
    theta   = reshape(draw_precision(rhs_th, P_theta), M, T);

    %% Sign flip (Fruhwirth-Schnatter & Wagner 2010) -------------------
    % Per coefficient j, jointly flip (s_j, theta_{j,:}) with probability
    % 1/2. The product s_j * theta_{j,t} is invariant so the posterior is
    % preserved; the move lets the chain visit both sign-equivalent
    % basins (s and theta are non-identified individually).
    sgn   = sign(rand(M, 1) - 0.5);
    s     = sgn .* s;
    theta = sgn .* theta;

    %% Block 4: Sigma (inverse-Wishart, full matrix) -------------------
    % Reuse y_bar_mat = Y - X*Psi_bar' from Block 3 (Psi_bar unchanged).
    sTheta3D = reshape(s .* theta, N, K, T);
    TV_part  = reshape(pagemtimes(sTheta3D, X_cols), N, T)';
    E_full   = y_bar_mat - TV_part;
    Sigma    = iwishrnd(priors.Sigma_S0 + E_full' * E_full, priors.Sigma_nu + T);

    %% store -----------------------------------------------------------
    if iter > opts.burn && mod(iter - opts.burn, opts.thin) == 0
        sIdx = sIdx + 1;
        draws.Psi_bar(sIdx, :, :)  = Psi_bar;
        draws.lambda(sIdx, :)      = [lambda_own, lambda_cross];
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
