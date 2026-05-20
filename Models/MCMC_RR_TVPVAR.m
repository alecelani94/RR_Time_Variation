function draws = MCMC_RR_TVPVAR(Data, P, R, priors, opts)
% MCMC_RR_TVPVAR   Gibbs sampler for the rank-R bilinear TVP-VAR(P).
%
% Model (non-centered, intercept folded into the bilinear block a la
% Chan-Strachan (2020, JoE)):
%
%   y_t = X_{t-1} psi_bar
%       + X_{t-1} * S_l * vec(A_t * B_t')
%       + eps_t,           eps_t ~ N(0, Sigma),
%
% where X_{t-1} = [1, y_{t-1}', ..., y_{t-P}']' has dim K = N*P + 1, so
%   psi_bar  = vec(Psi_bar)     with Psi_bar an N x K static coef matrix
%                                (first column = static intercept),
%   A_t   in R^(N x R),         B_t in R^(K x R)         (both TV, RW),
%   S_l   is the NK x NK diagonal loading matrix (one std per element).
%
% Renormalization:  V^a = (R*T)^(-1/2) I_N,  V^b = (R*T)^(-1/2) I_K.
%
% Sampler blocks (Gibbs):
%   1. Psi_bar  -- system-wise Gaussian (Minnesota prior, intercept fixed
%                  by lambda_const, lag coefs hierarchical via lambda_own
%                  / lambda_cross)
%   2. lambda_own, lambda_cross  -- hierarchical Minnesota shrinkages
%        (skipped if priors.lam_hier == false)
%   3. a  -- banded precision sampler
%   4. b  -- banded precision sampler
%   5. S_l_m  (N x K reshape of diag(S_l))  -- Gaussian conjugate
%   6. Sigma  -- inverse-Wishart (full matrix, NOT eq by eq)
%
% Inputs:
%   Data    : T_obs x N  (already pre-sliced by date)
%   P       : VAR lag order
%   R       : rank of the bilinear decomposition
%   priors  : struct with user-facing hyperparameters
%   opts    : struct -- burn, mcmc, thin, print_freq
%
% Output:
%   draws   : struct of MCMC draws (one field per parameter block)

%% ---- setup ------------------------------------------------------------

[T_obs, N] = size(Data);  %#ok<ASGLU>
K = N * P + 1;                              % includes intercept column

Y = Data(P+1:end, :);                       % T x N (LHS)
T = size(Y, 1);

% X: T x K  with row t = [1, y_{t-1}', y_{t-2}', ..., y_{t-P}']
X = ones(T, K);
for p = 1:P
    X(:, 1 + (p-1)*N + 1 : 1 + p*N) = Data(P+1-p : end-p, :);
end

K_lag = N * P;                              % lag-only column dim (for Minnesota)

%% ---- prior construction (built from user-facing hyperparameters) ------

% --- AR(4) per-equation: variances + full residual covariance ---
[priors.scales, priors.Sigma_AR] = AR_scales(Y, 4);

% --- Minnesota prior constants (intercept variance + lag scaling) ---
[priors.V_phi0, priors.C_Psi] = construct_Minnesota( ...
    priors.scales, P, priors.lam(1), priors.lam(4));

% --- own / cross-lag positions in vec(Psi_bar_lag), i.e. N x K_lag ---
own_mask = false(N, K_lag);
for l = 1:P
    for i = 1:N
        own_mask(i, (l-1)*N + i) = true;
    end
end
priors.ps_own   = find(own_mask(:));
priors.ps_cross = find(~own_mask(:));

% --- Gamma prior on lambda_own / lambda_cross (mean = shape * scale) ---
priors.par_lam = [priors.sh_lam, priors.lam(2)/priors.sh_lam;     % own
                  priors.sh_lam, priors.lam(3)/priors.sh_lam];    % cross

% --- Sigma ~ IW(nu, S0): centered at full AR(4) residual covariance ---
priors.Sigma_nu = N + priors.Sigma_nu_offset;
priors.Sigma_S0 = (priors.Sigma_nu - N - 1) * priors.Sigma_AR;

% --- Renormalized RW variance for a and b (symmetric, (RT)^(-1/2)) ---
priors.v_ab = (R * T)^(-1/2);

%% ---- precomputes (fixed across the Gibbs loop) ------------------------

XX = X' * X;                            % K x K  (used in Block 1)

%% ---- initialization ---------------------------------------------------

% Static coefficients: unrestricted OLS on the VAR(P) system.
Psi_bar      = (Y' * X) / (X' * X);          % N x K
E_ols        = Y - X * Psi_bar';
Sigma        = (E_ols' * E_ols) / (T - K);   % OLS residual covariance

% Hierarchical Minnesota lambdas: start at prior means.
lambda_own   = priors.lam(2);
lambda_cross = priors.lam(3);

% TV factors: simulate one path from the RW prior with renormalized
% innovation variance v_ab = (R*T)^(-1/2). Initial condition A_0 = B_0 = 0,
% so a_t = sum_{s<=t} u^a_s is built via cumsum.
sigma_ab     = sqrt(priors.v_ab);
A            = zeros(T, N, R);
B            = zeros(T, K, R);
for r = 1:R
    A(:, :, r) = cumsum(sigma_ab * randn(T, N), 1);
    B(:, :, r) = cumsum(sigma_ab * randn(T, K), 1);
end

% Loading S_l: draw each diagonal element from its mean-zero Gaussian
% prior N(0, V_sl).
S_l_m        = sqrt(priors.V_sl) * randn(N, K);

%% ---- storage ----------------------------------------------------------

M = opts.mcmc;
draws.Psi_bar      = zeros(M, N, K);
draws.lambda_own   = zeros(M, 1);
draws.lambda_cross = zeros(M, 1);
draws.A            = zeros(M, T, N, R);
draws.B            = zeros(M, T, K, R);
draws.S_l_m        = zeros(M, N, K);
draws.Sigma        = zeros(M, N, N);

%% ---- Gibbs loop -------------------------------------------------------

niter = opts.burn + M * opts.thin;

s = 0;
tic;
for iter = 1:niter

    %% Block 1: static coefficients Psi_bar ----------------------------
    % System-wise conjugate Gaussian draw of vec(Psi_bar) given the TV
    % component and the hierarchical Minnesota prior.

    % --- TV contribution to each y_t ---
    % Phi_TV(t) = S_l_m .* (A_t B_t'), then row t of TV_part is Phi_TV(t) * x_t
    TV_part = zeros(T, N);
    for t = 1:T
        A_t = reshape(A(t, :, :), N, R);
        B_t = reshape(B(t, :, :), K, R);
        TV_part(t, :) = ((S_l_m .* (A_t * B_t')) * X(t, :)')';
    end
    Y_bar_psi = Y - TV_part;                    % T x N residualized data

    % --- Prior precision diagonal of vec(Psi_bar), NK x 1, in column-stack order ---
    % First N entries: V_phi0 (intercept column, fixed by lambda_const).
    % Remaining N*K_lag entries: C_Psi multiplied by lambda_own / lambda_cross.
    V_lag_vec = priors.C_Psi(:);
    V_lag_vec(priors.ps_own)   = lambda_own   * V_lag_vec(priors.ps_own);
    V_lag_vec(priors.ps_cross) = lambda_cross * V_lag_vec(priors.ps_cross);
    V_psi_diag = [priors.V_phi0; V_lag_vec];    % NK x 1

    % --- Posterior precision and mean (X'X kron Sigma^{-1} + diag prior) ---
    Sigma_inv = Sigma \ eye(N);
    P_post    = spdiags(1 ./ V_psi_diag, 0, N*K, N*K) + kron(XX, Sigma_inv);
    rhs       = Sigma_inv * (Y_bar_psi' * X);   % N x K
    rhs       = rhs(:);                          % NK x 1

    % --- Cholesky draw ---
    L_post   = chol(P_post, 'lower');
    mu_post  = L_post' \ (L_post \ rhs);
    psi_draw = mu_post + L_post' \ randn(N*K, 1);
    Psi_bar  = reshape(psi_draw, N, K);

    %% Block 2: lambda_own, lambda_cross (hierarchical Minnesota) ------
    if priors.lam_hier
        % TODO: GIG draws via gigrnd given Psi_bar(:, 2:end), C_Psi,
        %       ps_own / ps_cross, and the Gamma hyperprior in priors.par_lam.
    end

    %% Block 3: a  (banded precision sampler) --------------------------
    % TODO: Z^a_t = X_{t-1} * S_l * (B_t kron I_N), stack, banded draw of a.

    %% Block 4: b  (banded precision sampler) --------------------------
    % TODO: Z^b_t = X_{t-1} * S_l * (I_K kron A_t), stack, banded draw of b.

    %% Block 5: S_l_m  (Gaussian conjugate on the loading) -------------
    % TODO: Gaussian conjugate eq-by-eq given A, B and residuals.

    %% Block 6: Sigma  (inverse-Wishart, full matrix) ------------------
    % Recompute TV contribution with the latest A, B, S_l_m, then form
    % the full residual E = Y - X*Psi_bar' - TV_part and draw Sigma.
    for t = 1:T
        A_t = reshape(A(t, :, :), N, R);
        B_t = reshape(B(t, :, :), K, R);
        TV_part(t, :) = ((S_l_m .* (A_t * B_t')) * X(t, :)')';
    end
    E_full = Y - X * Psi_bar' - TV_part;
    Sigma  = iwishrnd(priors.Sigma_S0 + E_full' * E_full, priors.Sigma_nu + T);

    %% store -----------------------------------------------------------
    if iter > opts.burn && mod(iter - opts.burn, opts.thin) == 0
        s = s + 1;
        draws.Psi_bar(s, :, :) = Psi_bar;
        draws.lambda_own(s)    = lambda_own;
        draws.lambda_cross(s)  = lambda_cross;
        draws.A(s, :, :, :)    = A;
        draws.B(s, :, :, :)    = B;
        draws.S_l_m(s, :, :)   = S_l_m;
        draws.Sigma(s, :, :)   = Sigma;
    end

    if mod(iter, opts.print_freq) == 0
        fprintf('Iter %d / %d  (elapsed %.1fs)\n', iter, niter, toc);
    end
end
fprintf('Total elapsed: %.2f sec\n', toc);

end
