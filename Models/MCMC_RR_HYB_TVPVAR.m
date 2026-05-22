function draws = MCMC_RR_HYB_TVPVAR(Data, P, R, priors, opts)
% MCMC_RR_HYB_TVPVAR   Gibbs sampler for the rank-R HYBRID bilinear
% TVP-VAR(P): intercept TV gets its own unrestricted RW (N states per
% period), and the rank-R bilinear A_t B_t' covers only the lag block.
% Bilinear factor a or b can be RW or AR(1) (selected via priors.ar_factor).
%
% Motivation: in the non-hybrid spec the bilinear factorization spends
% rank budget on intercept dynamics, which (a) have no natural low-rank
% cross-equation structure (they are per-equation idiosyncratic trends)
% and (b) dominate the TV signal as N grows. Lifting the intercept TV out
% of the bilinear restores rank to the lag block. See brainstorm doc Sec 3.
%
% Model (non-centered):
%
%   y_t = X_t * phi_bar  +  diag(s_0) * theta_0_t                       (intercept block)
%                       +  X_lag_t * S_l * vec(A_t * B_t')              (lag bilinear)
%                       +  eps_t,
%   theta_0_t   = theta_0_{t-1}                            + u_t^{0},   u_t^{0} ~ N(0, I_N)
%   A_t         = rho_a * A_{t-1} or A_{t-1}               + u_t^{a},   u_t^{a} ~ N(0, v_a*I_N)
%   B_t         = rho_b * B_{t-1} or B_{t-1}               + u_t^{b},   u_t^{b} ~ N(0, v_b*I_{K_lag})
%   eps_t       ~ N(0, Sigma),
%
% where  X_t      = x_t'     kron I_N,  x_t      = [1, y_{t-1}', ..., y_{t-P}']'
%        X_lag_t  = x_lag_t' kron I_N,  x_lag_t  = [y_{t-1}', ..., y_{t-P}']'
%        K        = N*P + 1,   K_lag = N*P,
%        M        = N*K,       M_lag = N*K_lag = M - N,
%        A_t      in R^{N      x R},  B_t in R^{K_lag x R},
%        S_l      = diag(vec(S_l_m)) with S_l_m an N x K_lag loading matrix,
%        s_0      in R^N (intercept loading; theta_0 in R^{N x T}).
%
% Bilinear law of motion (priors.ar_factor):
%   'none' (default) : both A and B are RW                  -> 2RW + hybrid
%   'A'              : A is AR(1) with persistence rho      -> RWAR + hybrid
%   'B'              : B is AR(1) with persistence rho      -> RWAR + hybrid
%
% Variance calibration (match-at-T, exact for any rho in (-1,1)):
%   v_RW = (R*T)^(-1/2)
%   v_AR = (1 - rho^2) / (1 - rho^(2T)) * sqrt(T/R)
%
% Sampler blocks (Gibbs, Chan 2023 JBES joint update + extended sign flips):
%   1.  (phi_bar, s_0, S_l_m)   -- JOINT Gaussian conjugate.
%   2.  lambda_own, lambda_cross -- hierarchical GIG (if priors.lam_hier).
%   3.  theta_0                  -- banded precision (bandwidth N, intercept RW).
%   4.  a                        -- banded precision (bandwidth NR).
%   5.  b                        -- banded precision (bandwidth K_lag * R).
%   5.5 sign flips               -- per-rank, per-equation (extended to include
%                                   intercept block), per-regressor on K_lag cols.
%   6.  Sigma                    -- inverse-Wishart (full matrix).

%% ---- setup ------------------------------------------------------------

[T_obs, N] = size(Data);  %#ok<ASGLU>
K       = N * P + 1;        % includes intercept column
K_lag   = N * P;            % lag-only columns
M       = N * K;
M_lag   = N * K_lag;        % = M - N

Y = Data(P+1:end, :);       % T x N
T = size(Y, 1);

% Full X (T x K): row t = [1, y_{t-1}', ..., y_{t-P}']
X = ones(T, K);
for p = 1:P
    X(:, 1 + (p-1)*N + 1 : 1 + p*N) = Data(P+1-p : end-p, :);
end
X_lag = X(:, 2:end);        % T x K_lag

NR     = N * R;
K_lagR = K_lag * R;

%% ---- prior construction -----------------------------------------------

[priors.scales, priors.Sigma_AR] = AR_scales(Y, 4);

% construct_Minnesota returns V_phi0 (N x 1, intercept) and C_Psi (N x K_lag).
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

priors.par_lam = [priors.sh_lam, priors.lam(2)/priors.sh_lam;
                  priors.sh_lam, priors.lam(3)/priors.sh_lam];

priors.Sigma_nu = N + priors.Sigma_nu_offset;
priors.Sigma_S0 = (priors.Sigma_nu - N - 1) * priors.Sigma_AR;

% --- Bilinear law-of-motion: RW default, switch one factor to AR(1) if asked.
priors.rho_a = 1;
priors.rho_b = 1;
priors.v_a   = (R * T)^(-1/2);
priors.v_b   = (R * T)^(-1/2);

if isfield(priors, 'ar_factor') && ~isempty(priors.ar_factor) ...
   && ~strcmpi(priors.ar_factor, 'none')

    if ~isfield(priors, 'rho') || isempty(priors.rho)
        error('MCMC_RR_HYB_TVPVAR:missingRho', ...
              'priors.rho must be set when priors.ar_factor is ''A'' or ''B''.');
    end

    v_AR = (1 - priors.rho^2) / (1 - priors.rho^(2*T)) * sqrt(T / R);
    switch upper(priors.ar_factor)
        case 'A'
            priors.rho_a = priors.rho;
            priors.v_a   = v_AR;
        case 'B'
            priors.rho_b = priors.rho;
            priors.v_b   = v_AR;
        otherwise
            error('MCMC_RR_HYB_TVPVAR:badAR', ...
                  'priors.ar_factor must be ''A'', ''B'', or ''none'' (got ''%s'').', ...
                  priors.ar_factor);
    end
end

% Per-element prior variance of the loading.
% Layout in mu's loading block: [s_0 (N entries); vec(S_l_m) (N*K_lag entries)].
V_sl_lag_vec = priors.V_sl_lag * ones(M_lag, 1);            % N*K_lag x 1
V_s0_vec     = priors.V_sl_intercept * ones(N, 1);          % N x 1
V_sl_vec     = [V_s0_vec; V_sl_lag_vec];                    % M x 1 (s_0 then S_l_m)

%% ---- precomputes (fixed across the Gibbs loop) ------------------------

% Reshapes of X (constant) reused inside the loop:
Xt          = X.';                         % K x T
X_broad     = reshape(Xt, 1, K, T);         % 1 x K x T

X_lagt      = X_lag.';                      % K_lag x T
X_lag_broad = reshape(X_lagt, 1, K_lag, T); % 1 x K_lag x T
X_lag_cols  = reshape(X_lagt, K_lag, 1, T); % K_lag x 1 x T
X_lag_4D    = reshape(X_lagt, 1, 1, K_lag, T); % 1 x 1 x K_lag x T (for b block)

% Time-ordered y stack (used in joint Block 1, theta_0 Block 3, a/b Blocks 4/5)
y_full = reshape(Y', N*T, 1);

% Static block of the joint design has CONSTANT entries X(t,k); precompute.
vv_Zs_phi = reshape(repmat(X_broad, N, 1, 1), [], 1);    % N*K*T x 1

% --- First-difference / AR(1) operators (fixed across iterations).
%     Sub-diagonal entry is -1 for RW, -rho for AR(1). theta_0 is always RW.
e1     = ones(T, 1);
H_T_a  = spdiags([-priors.rho_a * e1, e1], [-1, 0], T, T);
H_T_b  = spdiags([-priors.rho_b * e1, e1], [-1, 0], T, T);
H_T_0  = spdiags([-              e1, e1], [-1, 0], T, T);

HtH_a   = (kron(H_T_a, speye(NR    )))' * kron(H_T_a, speye(NR    ));
HtH_b   = (kron(H_T_b, speye(K_lagR)))' * kron(H_T_b, speye(K_lagR));
HtH_th0 = (kron(H_T_0, speye(N     )))' * kron(H_T_0, speye(N     ));

% --- Sparse-triplet (row, col) indices reused each iteration ---
% Convention: linearization (i, ..., t) with i innermost matches our
% reshape(..., (:)) flatten order so values can be stuffed in directly.

% (1) phi_bar joint block: K columns of the static design
idx_v       = (0:(N*K*T - 1))';
i_v         = mod(idx_v, N);
k_v         = mod(floor(idx_v / N), K);
t_v         = floor(idx_v / (N*K));
ii_V_pre    = t_v * N + i_v + 1;       % row in NT
jj_V_pre    = k_v * N + i_v + 1;       % col in M (phi_bar block, no t-shift)

% (2) s_0 part of the loading block: only N values per t, at cols M+1..M+N
idx_s0      = (0:(N*T - 1))';
i_s0        = mod(idx_s0, N);
t_s0        = floor(idx_s0 / N);
ii_s0_pre   = t_s0 * N + i_s0 + 1;
jj_s0_pre   = i_s0 + 1 + M;            % shift into s_0 block (M+1..M+N)

% (3) S_l_m part of the loading block: N*K_lag values per t, at cols M+N+1..2M
idx_vl      = (0:(N*K_lag*T - 1))';
i_vl        = mod(idx_vl, N);
k_vl        = mod(floor(idx_vl / N), K_lag);
t_vl        = floor(idx_vl / (N*K_lag));
ii_vl_pre   = t_vl * N + i_vl + 1;
jj_vl_pre   = k_vl * N + i_vl + 1 + M + N;   % shift into S_l_m block (M+N+1..2M)

% --- Combined joint indices: [phi_bar (size N*K*T); s_0 (N*T); s_l (N*K_lag*T)]
ii_joint    = [ii_V_pre; ii_s0_pre; ii_vl_pre];
jj_joint    = [jj_V_pre; jj_s0_pre; jj_vl_pre];

% (4) a-block design Z_a: NT x NR*T (same shape as non-hybrid; bandwidth NR)
idx_a    = (0:(N*R*T - 1))';
i_a      = mod(idx_a, N);
r_a      = mod(floor(idx_a / N), R);
t_a      = floor(idx_a / (N*R));
ii_a_pre = t_a * N  + i_a + 1;
jj_a_pre = t_a * NR + r_a * N + i_a + 1;

% (5) b-block design Z_b: NT x K_lagR*T (one fewer column block than non-hybrid)
idx_b    = (0:(N*R*K_lag*T - 1))';
i_b      = mod(idx_b, N);
r_b      = mod(floor(idx_b / N),         R);
k_b      = mod(floor(idx_b / (N*R)),     K_lag);
t_b      = floor(idx_b / (N*R*K_lag));
ii_b_pre = t_b * N      + i_b + 1;
jj_b_pre = t_b * K_lagR + k_b * R + r_b + 1;

clear idx_v i_v k_v t_v idx_s0 i_s0 t_s0 idx_vl i_vl k_vl t_vl ...
      idx_a i_a r_a t_a idx_b i_b r_b k_b t_b

% (6) theta_0 design: time-block-diagonal with diag(s_0); precompute its
% sparse indices. The diagonal indexing is (1:N*T, 1:N*T).
ii_th0 = (1:N*T)';
jj_th0 = (1:N*T)';

% GIG hyperparameters for the hierarchical Minnesota update (fixed)
p_own_gig     = priors.par_lam(1,1) - numel(priors.ps_own)   / 2;
p_cross_gig   = priors.par_lam(2,1) - numel(priors.ps_cross) / 2;
psi_own_gig   = 2 / priors.par_lam(1,2);
psi_cross_gig = 2 / priors.par_lam(2,2);

%% ---- initialization ---------------------------------------------------

% Static + Sigma: OLS warm start (using full X, intercept included).
Psi_bar = (Y' * X) / (X' * X);                % N x K
E_ols   = Y - X * Psi_bar';
Sigma   = (E_ols' * E_ols) / (T - K);

% Hierarchical lambdas: start at prior means.
lambda_own   = priors.lam(2);
lambda_cross = priors.lam(3);

% Bilinear TV factors: simulate one path from each factor's prior.
A = filter(1, [1, -priors.rho_a], sqrt(priors.v_a) * randn(T, N,     R), [], 1);
B = filter(1, [1, -priors.rho_b], sqrt(priors.v_b) * randn(T, K_lag, R), [], 1);

% Intercept TV state: simulate a unit-variance RW path (theta_0_0 = 0).
theta_0 = cumsum(randn(N, T), 2);             % N x T

% (s_0, S_l_m) sampled jointly with Psi_bar in Block 1; no init needed.

%% ---- storage ----------------------------------------------------------

Mc = opts.mcmc;
draws.Psi_bar = zeros(Mc, N, K);
draws.lambda  = zeros(Mc, 2);
draws.A       = zeros(Mc, T, N,     R);
draws.B       = zeros(Mc, T, K_lag, R);
draws.S_l_m   = zeros(Mc, N, K_lag);
draws.s_0     = zeros(Mc, N);
draws.theta_0 = zeros(Mc, N, T);
draws.Sigma   = zeros(Mc, N, N);

%% ---- Gibbs loop -------------------------------------------------------

niter = opts.burn + Mc * opts.thin;
sIdx  = 0;
tic;
for iter = 1:niter

    %% Block 1 (JOINT): phi_bar (full M), s_0 (N), S_l_m (M_lag) ----------
    % Observation:
    %   y_t = X_t * phi_bar
    %       + diag(theta_0_t) * s_0                     (intercept TV piece)
    %       + X_lag_t * diag(vec(A_t B_t')) * vec(S_l_m) (lag bilinear piece)
    %       + eps_t
    % Stack mu = [phi_bar; s_0; vec(S_l_m)] of dim M + N + M_lag = 2M.

    % --- AB_3D = A_t * B_t' for all t (N x K_lag x T) ---
    A_pages = permute(A, [2 3 1]);                       % N x R x T
    B_pages = permute(B, [2 3 1]);                       % K_lag x R x T
    AB_3D   = pagemtimes(A_pages, 'none', B_pages, 'transpose');  % N x K_lag x T

    % --- Joint design values ---
    vv_S0   = reshape(theta_0, [], 1);                   % N*T x 1 (column-major)
    vv_W    = reshape(AB_3D .* X_lag_broad, [], 1);      % N*K_lag*T x 1
    Z_tilde = sparse(ii_joint, jj_joint, ...
                     [vv_Zs_phi; vv_S0; vv_W], N*T, 2*M);

    % --- Joint prior diagonal (2M x 1) ---
    %   phi_bar block: V_phi0 (intercept) + scaled C_Psi (lag).
    V_lag_vec = priors.C_Psi(:);
    V_lag_vec(priors.ps_own)   = lambda_own   * V_lag_vec(priors.ps_own);
    V_lag_vec(priors.ps_cross) = lambda_cross * V_lag_vec(priors.ps_cross);
    V_psi_diag = [priors.V_phi0; V_lag_vec];             % M x 1

    V_mu_diag = [V_psi_diag; V_sl_vec];                  % 2M x 1

    % --- Sigma^-1 broadcast over t ---
    Sigma_inv   = Sigma \ eye(N);
    Sigma_inv_T = kron(speye(T), Sigma_inv);

    % --- Posterior precision (sparse + dense 2M x 2M) and rhs ---
    P_post = spdiags(1 ./ V_mu_diag, 0, 2*M, 2*M) ...
           + Z_tilde' * Sigma_inv_T * Z_tilde;
    rhs    = Z_tilde' * (Sigma_inv_T * y_full);

    mu      = draw_precision(rhs, P_post);
    Psi_bar = reshape(mu(1:M),         N, K);
    s_0     = mu(M+1 : M+N);
    S_l_m   = reshape(mu(M+N+1 : 2*M), N, K_lag);

    %% Block 2: lambda_own, lambda_cross (hierarchical Minnesota) --------
    if priors.lam_hier
        Psi_lag      = Psi_bar(:, 2:end);
        sq_std_full  = Psi_lag(:).^2 ./ priors.C_Psi(:);
        lambda_own   = gigrnd(p_own_gig,   psi_own_gig, ...
                              sum(sq_std_full(priors.ps_own)),   1);
        lambda_cross = gigrnd(p_cross_gig, psi_cross_gig, ...
                              sum(sq_std_full(priors.ps_cross)), 1);
    end

    %% Residualize on the new static block (reused below) ----------------
    y_bar_mat = Y - X * Psi_bar';                        % T x N
    y_bar_vec = reshape(y_bar_mat', N*T, 1);

    %% Block 3: theta_0 (banded precision, bandwidth N) ------------------
    % Sub-residualize on the lag bilinear (theta_0 sees only the intercept TV):
    %   r_t = y_t - X_t*phi_bar - X_lag_t * (vec(A_t B_t') .* vec(S_l_m))
    %       = diag(s_0) * theta_0_t + eps_t
    % Z_th0 is block-diagonal with diag(s_0) repeated T times, so
    %   Z' * (I_T kron Sigma^-1) * Z = I_T kron ((s_0 s_0') .* Sigma^-1).
    M_3D_lag = S_l_m .* AB_3D;                            % N x K_lag x T
    TV_lag   = reshape(pagemtimes(M_3D_lag, X_lag_cols), N, T)';   % T x N
    r_th0_mat = y_bar_mat - TV_lag;                       % T x N
    r_th0_vec = reshape(r_th0_mat', N*T, 1);

    block_S0_Sinv_S0 = (s_0 * s_0.') .* Sigma_inv;        % N x N
    P_th0 = HtH_th0 + kron(speye(T), block_S0_Sinv_S0);
    % rhs_th0_t = diag(s_0) * Sigma^-1 * r_t  ->  s_0 .* (Sigma^-1 * R_mat) per t
    rhs_th0_mat = s_0 .* (Sigma_inv * r_th0_mat.');       % N x T
    rhs_th0     = rhs_th0_mat(:);
    th0_draw    = draw_precision(rhs_th0, P_th0);
    theta_0     = reshape(th0_draw, N, T);

    %% Block 4: a (banded precision, bandwidth NR) -----------------------
    % a sees the residual AFTER subtracting both static AND the intercept TV piece.
    intTV_mat   = (s_0 .* theta_0).';                     % T x N  (= theta_0_t .* s_0 row-wise)
    y_bar_a_mat = y_bar_mat - intTV_mat;                  % T x N
    y_bar_a_vec = reshape(y_bar_a_mat', N*T, 1);

    SlX_pages = S_l_m .* X_lag_broad;                     % N x K_lag x T
    SXB_3D    = pagemtimes(SlX_pages, B_pages);            % N x R x T
    Z_a       = sparse(ii_a_pre, jj_a_pre, SXB_3D(:), N*T, NR*T);

    P_a    = (1/priors.v_a) * HtH_a + Z_a' * Sigma_inv_T * Z_a;
    rhs_a  = Z_a' * (Sigma_inv_T * y_bar_a_vec);
    a_draw = draw_precision(rhs_a, P_a);
    A      = permute(reshape(a_draw, N, R, T), [3 1 2]);  % T x N x R

    %% Block 5: b (banded precision, bandwidth K_lag*R) ------------------
    A_4D = reshape(permute(A, [2 3 1]), N, R, 1, T);      % N x R x 1 x T
    S_4D = reshape(S_l_m, N, 1, K_lag, 1);                 % N x 1 x K_lag x 1
    vv_b = reshape(A_4D .* S_4D .* X_lag_4D, [], 1);       % N*R*K_lag*T x 1
    Z_b  = sparse(ii_b_pre, jj_b_pre, vv_b, N*T, K_lagR*T);

    P_b    = (1/priors.v_b) * HtH_b + Z_b' * Sigma_inv_T * Z_b;
    rhs_b  = Z_b' * (Sigma_inv_T * y_bar_a_vec);
    b_draw = draw_precision(rhs_b, P_b);
    % jj_b_pre orders (k, r) with r innermost (= vec(B_t')), so reshape as
    % (R, K_lag, T) and permute to (T, K_lag, R).
    B      = permute(reshape(b_draw, R, K_lag, T), [3 2 1]);  % T x K_lag x R

    %% Sign flips -- N + K_lag + R independent posterior-preserving moves
    % (a) Per-rank R: flip (A(:,:,r), B(:,:,r)).
    sgn_R = sign(rand(R, 1) - 0.5);
    for r = 1:R
        A(:, :, r) = sgn_R(r) * A(:, :, r);
        B(:, :, r) = sgn_R(r) * B(:, :, r);
    end

    % (b) Per-equation N: flip (S_l_m(i,:), A(:,i,:), s_0(i), theta_0(i,:)).
    sgn_N = sign(rand(N, 1) - 0.5);
    S_l_m   = sgn_N .* S_l_m;                              % flip row i of S_l_m
    A       = A .* reshape(sgn_N, 1, N, 1);                 % flip row i of A across t,r
    s_0     = sgn_N .* s_0;                                 % flip i-th intercept loading
    theta_0 = sgn_N .* theta_0;                             % flip i-th row of theta_0 across t

    % (c) Per-regressor K_lag: flip (S_l_m(:,k), B(:,k,:)).
    sgn_Kl = sign(rand(K_lag, 1) - 0.5);
    S_l_m = S_l_m .* sgn_Kl(:)';                            % flip col k of S_l_m
    B     = B     .* reshape(sgn_Kl, 1, K_lag, 1);          % flip k-th row of B across t,r

    %% Block 6: Sigma (inverse-Wishart, full matrix) ---------------------
    % Recompute AB_3D with the latest (A, B); reuse y_bar_mat for residuals.
    A_pages  = permute(A, [2 3 1]);
    B_pages  = permute(B, [2 3 1]);
    AB_3D    = pagemtimes(A_pages, 'none', B_pages, 'transpose');
    M_3D_lag = S_l_m .* AB_3D;
    TV_lag   = reshape(pagemtimes(M_3D_lag, X_lag_cols), N, T)';   % T x N
    intTV    = (s_0 .* theta_0).';                                   % T x N
    E_full   = y_bar_mat - intTV - TV_lag;
    Sigma    = iwishrnd(priors.Sigma_S0 + E_full' * E_full, ...
                        priors.Sigma_nu + T);

    %% store -----------------------------------------------------------
    if iter > opts.burn && mod(iter - opts.burn, opts.thin) == 0
        sIdx = sIdx + 1;
        draws.Psi_bar(sIdx, :, :)  = Psi_bar;
        draws.lambda(sIdx, :)      = [lambda_own, lambda_cross];
        draws.A(sIdx, :, :, :)     = A;
        draws.B(sIdx, :, :, :)     = B;
        draws.S_l_m(sIdx, :, :)    = S_l_m;
        draws.s_0(sIdx, :)         = s_0';
        draws.theta_0(sIdx, :, :)  = theta_0;
        draws.Sigma(sIdx, :, :)    = Sigma;
    end

    if mod(iter, opts.print_freq) == 0
        fprintf('Iter %d / %d  (elapsed %.1fs)\n', iter, niter, toc);
    end
end
fprintf('Total elapsed: %.2f sec\n', toc);

end
