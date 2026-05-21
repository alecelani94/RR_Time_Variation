function draws = MCMC_RR_TVPVAR_v2(Data, P, R, priors, opts)
% MCMC_RR_TVPVAR_v2  Gibbs sampler for the rank-R bilinear TVP-VAR(P) using
% the b-block convention b_t = vec(B_t) (standard column-major), as in the
% formulation suggested by Gemini.
%
% Algebraically equivalent to MCMC_RR_TVPVAR (v1): both samplers target the
% same posterior over (Psi_bar, A_t, B_t, S_l_m, Sigma, lambdas) and should
% produce identical posterior summaries (to MCMC error). The only
% difference is the internal ordering of the b_t vector and the
% corresponding column ordering of the Z_b design matrix:
%
%     v1 (MCMC_RR_TVPVAR):  b_t := vec(B_t')  -- r innermost within (k, r)
%                            ==> vec(A_t B_t') = (I_K kron A_t) b_t
%     v2 (this file):       b_t := vec(B_t)   -- k innermost within (k, r)
%                            ==> vec(A_t B_t') = (I_K kron A_t) K_{K,R} b_t
%                                where K_{K,R} is the commutation matrix
%                                (implemented here implicitly via column
%                                permutation of Z_b -- no explicit K_{K,R}).
%
% Use this file as a sanity check for the v1 benchmark. At R = 1 the two
% are bit-for-bit identical; for R >= 2 they should agree to MC error.

%% ---- setup ------------------------------------------------------------

[T_obs, N] = size(Data);  %#ok<ASGLU>
K = N * P + 1;
M = N * K;

Y = Data(P+1:end, :);                       % T x N
T = size(Y, 1);

% X: T x K  with row t = [1, y_{t-1}', y_{t-2}', ..., y_{t-P}']
X = ones(T, K);
for p = 1:P
    X(:, 1 + (p-1)*N + 1 : 1 + p*N) = Data(P+1-p : end-p, :);
end

K_lag = N * P;
NR    = N * R;
KR    = K * R;

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

priors.v_ab = (R * T)^(-1/2);

% Per-element prior variance of the matrix loading S_l_m (N x K).
% Col 1 is the intercept loading, cols 2..K are lag loadings.
V_sl_mat = [priors.V_sl_intercept * ones(N, 1), ...
            priors.V_sl_lag       * ones(N, K_lag)];
V_sl_vec = V_sl_mat(:);

%% ---- precomputes (fixed across the Gibbs loop) ------------------------

% Reshapes of X (constant) reused inside the loop:
Xt      = X.';                          % K x T
X_broad = reshape(Xt, 1, K, T);          % 1 x K x T  (broadcast over i)
X_cols  = reshape(Xt, K, 1, T);          % K x 1 x T  (pagemtimes col-vectors)
X_4D    = reshape(Xt, 1, 1, K, T);       % 1 x 1 x K x T (Block 4 broadcasting)

% Time-ordered y stack (used in joint Block 1, a Block 3, b Block 4)
y_full = reshape(Y', N*T, 1);

% Static block of the joint design has CONSTANT entries X(t,k); precompute.
vv_Zs = reshape(repmat(X_broad, N, 1, 1), [], 1);    % N*K*T x 1

% First-difference operators (fixed across iterations)
e1     = ones(T, 1);
H_T    = spdiags([-e1, e1], [-1, 0], T, T);
HtH_a  = (kron(H_T, speye(NR)))'  * kron(H_T, speye(NR));
HtH_b  = (kron(H_T, speye(KR)))'  * kron(H_T, speye(KR));

% Sparse-triplet (row, col) indices reused each iteration.
% Convention: linearization (i, ..., t) with i innermost matches our 3D/4D
% reshape(..., (:)) flatten order so values can be stuffed in directly.

% --- a-block design Z_a: NT x NR*T, K nonzeros per N rows per t ---
idx_a    = (0:(N*R*T - 1))';
i_a      = mod(idx_a, N);
r_a      = mod(floor(idx_a / N), R);
t_a      = floor(idx_a / (N*R));
ii_a_pre = t_a * N  + i_a + 1;
jj_a_pre = t_a * NR + r_a * N + i_a + 1;

% --- b-block design Z_b: NT x KR*T ---
idx_b    = (0:(N*R*K*T - 1))';
i_b      = mod(idx_b, N);
r_b      = mod(floor(idx_b / N),     R);
k_b      = mod(floor(idx_b / (N*R)), K);
t_b      = floor(idx_b / (N*R*K));
ii_b_pre = t_b * N  + i_b + 1;
jj_b_pre = t_b * KR + r_b * K + k_b + 1;   % v2: k innermost (matches vec(B_t))

% --- Common (i, k, t) -> NT-stack / M-stack indices used by joint block ---
idx_v    = (0:(N*K*T - 1))';
i_v      = mod(idx_v, N);
k_v      = mod(floor(idx_v / N), K);
t_v      = floor(idx_v / (N*K));
ii_V_pre = t_v * N + i_v + 1;       % row in NT
jj_V_pre = k_v * N + i_v + 1;       % col in M (no t-shift)
clear idx_a i_a r_a t_a idx_b i_b r_b k_b t_b idx_v i_v k_v t_v

% Doubled (row, col) indices for the joint design [Z_s, W], NT x 2M
ii_joint = [ii_V_pre; ii_V_pre];
jj_joint = [jj_V_pre; jj_V_pre + M];

% GIG hyperparameters for the hierarchical Minnesota update (fixed)
p_own_gig     = priors.par_lam(1,1) - numel(priors.ps_own)   / 2;
p_cross_gig   = priors.par_lam(2,1) - numel(priors.ps_cross) / 2;
psi_own_gig   = 2 / priors.par_lam(1,2);
psi_cross_gig = 2 / priors.par_lam(2,2);

%% ---- initialization ---------------------------------------------------

% Static + Sigma: OLS warm start.
Psi_bar = (Y' * X) / (X' * X);                % N x K
E_ols   = Y - X * Psi_bar';
Sigma   = (E_ols' * E_ols) / (T - K);

% Hierarchical lambdas: start at prior means.
lambda_own   = priors.lam(2);
lambda_cross = priors.lam(3);

% TV factors A, B: simulate from the renormalized RW prior.
% Used to seed AB_3D in the very first joint draw.
sigma_ab = sqrt(priors.v_ab);
A = cumsum(sigma_ab * randn(T, N, R), 1);     % T x N x R
B = cumsum(sigma_ab * randn(T, K, R), 1);     % T x K x R

% S_l_m sampled jointly with Psi_bar in Block 1; no init needed.

%% ---- storage ----------------------------------------------------------

Mc = opts.mcmc;
draws.Psi_bar = zeros(Mc, N, K);
draws.lambda  = zeros(Mc, 2);                 % [own, cross]
draws.A       = zeros(Mc, T, N, R);
draws.B       = zeros(Mc, T, K, R);
draws.S_l_m   = zeros(Mc, N, K);
draws.Sigma   = zeros(Mc, N, N);

%% ---- Gibbs loop -------------------------------------------------------

niter = opts.burn + Mc * opts.thin;
sIdx  = 0;
tic;
for iter = 1:niter

    %% Block 1 (JOINT): static phi_bar AND matrix loading S_l_m ---------
    % Substituting tau_t := vec(A_t B_t') and applying the same identity as
    % in MCMC_TVPVAR:
    %   y_t = X_t * phi_bar + X_t * diag(tau_t) * vec(S_l_m) + eps_t
    %       = Z_tilde_t * mu + eps_t,  mu := (phi_bar; vec(S_l_m)) in R^{2M}.
    %
    % Z_tilde_t = [X_t,  X_t * diag(tau_t)]  has the same sparsity pattern as
    % the unrestricted (Z_s, W) joint design, with tau_t replacing theta_t.

    % --- Compute AB_3D = A_t B_t' for all t via batched matmul ---
    A_pages = permute(A, [2 3 1]);                       % N x R x T
    B_pages = permute(B, [2 3 1]);                       % K x R x T
    AB_3D   = pagemtimes(A_pages, 'none', B_pages, 'transpose');  % N x K x T

    % --- Loading-block sparse values: vv_W(i,k,t) = AB(i,k,t) * X(t,k) ---
    vv_W    = reshape(AB_3D .* X_broad, [], 1);          % N*K*T x 1
    Z_tilde = sparse(ii_joint, jj_joint, ...
                     [vv_Zs; vv_W], N*T, 2*M);

    % --- Joint prior diagonal (2M x 1): Minnesota on phi, V_sl_vec on S_l ---
    V_lag_vec = priors.C_Psi(:);
    V_lag_vec(priors.ps_own)   = lambda_own   * V_lag_vec(priors.ps_own);
    V_lag_vec(priors.ps_cross) = lambda_cross * V_lag_vec(priors.ps_cross);
    V_psi_diag = [priors.V_phi0; V_lag_vec];
    V_mu_diag  = [V_psi_diag; V_sl_vec];

    % --- Sigma^-1 and block-diag over t (reused in Blocks 3, 4) ---
    Sigma_inv   = Sigma \ eye(N);
    Sigma_inv_T = kron(speye(T), Sigma_inv);

    % --- Posterior precision (dense 2M x 2M) and rhs ---
    P_post = spdiags(1 ./ V_mu_diag, 0, 2*M, 2*M) ...
           + Z_tilde' * Sigma_inv_T * Z_tilde;
    rhs    = Z_tilde' * (Sigma_inv_T * y_full);

    mu      = draw_precision(rhs, P_post);
    Psi_bar = reshape(mu(1:M),   N, K);
    S_l_m   = reshape(mu(M+1:end), N, K);

    %% Block 2: lambda_own, lambda_cross (hierarchical Minnesota) ------
    if priors.lam_hier
        Psi_lag      = Psi_bar(:, 2:end);
        sq_std_full  = Psi_lag(:).^2 ./ priors.C_Psi(:);
        lambda_own   = gigrnd(p_own_gig,   psi_own_gig, ...
                              sum(sq_std_full(priors.ps_own)),   1);
        lambda_cross = gigrnd(p_cross_gig, psi_cross_gig, ...
                              sum(sq_std_full(priors.ps_cross)), 1);
    end

    % --- Residualize on the new static block (reused in Blocks 3, 4, 5) ---
    y_bar_mat = Y - X * Psi_bar';                       % T x N
    y_bar_vec = reshape(y_bar_mat', N*T, 1);

    %% Block 3: a  (banded precision) ----------------------------------
    % Z^a_t column ((r-1)*N + i) has a single nonzero at row i with value
    % SXB(i, r, t) = sum_k (S_l_m(i,k) * X(t,k)) * B_t(k, r).
    SlX_pages = S_l_m .* X_broad;                       % N x K x T
    SXB_3D    = pagemtimes(SlX_pages, B_pages);          % N x R x T (B_pages reused)
    Z_a       = sparse(ii_a_pre, jj_a_pre, SXB_3D(:), N*T, NR*T);

    P_a    = (1/priors.v_ab) * HtH_a + Z_a' * Sigma_inv_T * Z_a;
    rhs_a  = Z_a' * (Sigma_inv_T * y_bar_vec);
    a_draw = draw_precision(rhs_a, P_a);
    A      = permute(reshape(a_draw, N, R, T), [3 1 2]);  % T x N x R

    %% Block 4: b  (banded precision) ----------------------------------
    % Z^b_t column ((k-1)*R + r) at row i has value X(t,k) * S_l_m(i,k) * A_t(i,r).
    A_4D = reshape(permute(A, [2 3 1]), N, R, 1, T);     % N x R x 1 x T
    S_4D = reshape(S_l_m, N, 1, K, 1);                    % N x 1 x K x 1
    vv_b = reshape(A_4D .* S_4D .* X_4D, [], 1);          % N*R*K*T x 1
    Z_b  = sparse(ii_b_pre, jj_b_pre, vv_b, N*T, KR*T);

    P_b    = (1/priors.v_ab) * HtH_b + Z_b' * Sigma_inv_T * Z_b;
    rhs_b  = Z_b' * (Sigma_inv_T * y_bar_vec);
    b_draw = draw_precision(rhs_b, P_b);
    % v2: jj_b_pre orders (k, r) with k innermost (= vec(B_t)), so the
    % matching column-major reshape is (K, R, T); then permute to (T, K, R).
    B      = permute(reshape(b_draw, K, R, T), [3 1 2]);  % T x K x R

    %% Sign flip (Fruhwirth-Schnatter & Wagner 2010) -------------------
    % Per rank component r, jointly flip (A(:,:,r), B(:,:,r)) w.p. 1/2;
    % A_t B_t' is invariant so the posterior is preserved.
    sgn_R = sign(rand(R, 1) - 0.5);
    for r = 1:R
        A(:, :, r) = sgn_R(r) * A(:, :, r);
        B(:, :, r) = sgn_R(r) * B(:, :, r);
    end

    %% Block 5: Sigma (inverse-Wishart, full matrix) -------------------
    % Recompute AB_3D with the latest (A, B); reuse y_bar_mat for residuals.
    A_pages = permute(A, [2 3 1]);
    B_pages = permute(B, [2 3 1]);
    AB_3D   = pagemtimes(A_pages, 'none', B_pages, 'transpose');
    M_3D    = S_l_m .* AB_3D;                            % N x K x T
    TV_part = reshape(pagemtimes(M_3D, X_cols), N, T)';   % T x N
    E_full  = y_bar_mat - TV_part;
    Sigma   = iwishrnd(priors.Sigma_S0 + E_full' * E_full, priors.Sigma_nu + T);

    %% store -----------------------------------------------------------
    if iter > opts.burn && mod(iter - opts.burn, opts.thin) == 0
        sIdx = sIdx + 1;
        draws.Psi_bar(sIdx, :, :)  = Psi_bar;
        draws.lambda(sIdx, :)      = [lambda_own, lambda_cross];
        draws.A(sIdx, :, :, :)     = A;
        draws.B(sIdx, :, :, :)     = B;
        draws.S_l_m(sIdx, :, :)    = S_l_m;
        draws.Sigma(sIdx, :, :)    = Sigma;
    end

    if mod(iter, opts.print_freq) == 0
        fprintf('Iter %d / %d  (elapsed %.1fs)\n', iter, niter, toc);
    end
end
fprintf('Total elapsed: %.2f sec\n', toc);

end
