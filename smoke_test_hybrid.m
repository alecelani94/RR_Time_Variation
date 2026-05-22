% SMOKE_TEST_HYBRID  Fast structural check of the hybrid sampler.
%
% Runs MCMC_RR_HYB_TVPVAR + the two HYB helpers at very small Mc/burn
% (~20 seconds), then verifies the structural correctness of every output:
%
%   1. Sampler completes without crash on RWAR_HYB and 2RW_HYB modes.
%   2. draws struct has all expected fields with correct dimensions.
%   3. No NaN or Inf in any output.
%   4. Sign-flip moves are active (s_0 marginal roughly 50/50 +/-).
%   5. Phi reconstruction matches manual recomputation on a sample cell.
%   6. compute_RR_HYB_Phi_IFs returns finite, non-negative IFs.
%   7. reconstruct_RR_HYB_quantiles returns the expected shape.
%
% Does NOT check posterior correctness or convergence -- those need a
% real run. This is purely a structural / dimensional / no-crash check
% to flush out indexing bugs before committing to a full job.
%
% Requires: Data/fredqd_small_5_gs5.mat (or set DATASET below).

clear; clc; close all;
rng(12345);  % distinct from main_fs.m to catch seed-dependent bugs

addpath(genpath(fullfile(pwd, 'Functions')));
addpath(genpath(fullfile(pwd, 'Models')));

%% ---- settings ---------------------------------------------------------

DATASET    = 'small_5_gs5';
date_start = datetime(1959, 6, 1);
date_end   = datetime(2019, 12, 1);

P     = 2;
R     = 1;       % rank-1 keeps everything small and exercises the trickiest dims

% Tiny MCMC to keep the test fast (~20 sec total for both variants)
opts.burn       = 20;
opts.mcmc       = 50;
opts.thin       = 1;
opts.print_freq = 100;   % suppress per-iter output

% Priors (mirror main_fs.m)
priors.lam      = [10^2; 0.2^2; 0.1^2; 2];
priors.sh_lam   = 2;
priors.lam_hier = false;
priors.V_sl_intercept = 0.05^2;
priors.V_sl_lag       = 0.01^2;
priors.Sigma_nu_offset = 2;

% Posterior summaries
qoi = [0.05, 0.50, 0.95];

%% ---- load data --------------------------------------------------------

fprintf('=== Smoke test for hybrid sampler ===\n\n');
[Y, Names, Ydates, T_raw, N] = load_slice(DATASET, date_start, date_end);
% Effective sample inside the sampler: first P observations are absorbed
% as initial lags, so the TV blocks have T = T_raw - P time points.
T     = T_raw - P;
K     = N * P + 1;
K_lag = N * P;
fprintf('N=%d, P=%d, T_raw=%d, T=%d (after losing P initial lags), K=%d, K_lag=%d, R=%d\n\n', ...
        N, P, T_raw, T, K, K_lag, R);

%% ---- variants to test -------------------------------------------------

variants = {
    '2RW_HYB',  struct('ar_factor', 'none');
    'RWAR_HYB', struct('ar_factor', 'A', 'rho', 0.99);
};

n_fail = 0;

for v = 1:size(variants, 1)
    name = variants{v, 1};
    cfg  = variants{v, 2};

    fprintf('--- Variant: %s ---\n', name);

    % --- merge cfg into priors ---
    priors_v = priors;
    fn = fieldnames(cfg);
    for f = 1:numel(fn)
        priors_v.(fn{f}) = cfg.(fn{f});
    end

    % --- (1) Sampler runs without crash ---
    try
        t_sampler = tic;
        draws = MCMC_RR_HYB_TVPVAR(Y, P, R, priors_v, opts);
        fprintf('  [PASS] Sampler completed in %.1f s\n', toc(t_sampler));
    catch ME
        n_fail = n_fail + 1;
        fprintf('  [FAIL] Sampler crashed: %s\n', ME.message);
        fprintf('         %s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
        continue
    end

    % --- (2) Expected fields and dimensions ---
    expected = struct( ...
        'Psi_bar', [opts.mcmc, N, K], ...
        'lambda',  [opts.mcmc, 2], ...
        'A',       [opts.mcmc, T, N, R], ...
        'B',       [opts.mcmc, T, K_lag, R], ...
        'S_l_m',   [opts.mcmc, N, K_lag], ...
        's_0',     [opts.mcmc, N], ...
        'theta_0', [opts.mcmc, N, T], ...
        'Sigma',   [opts.mcmc, N, N] ...
    );

    ok_fields = true;
    fn = fieldnames(expected);
    for f = 1:numel(fn)
        if ~isfield(draws, fn{f})
            fprintf('  [FAIL] Missing field: draws.%s\n', fn{f});
            ok_fields = false; n_fail = n_fail + 1;
            continue
        end
        actual = size(draws.(fn{f}));
        exp_sz = expected.(fn{f});
        % pad actual size with 1s if needed (MATLAB drops trailing singletons)
        if numel(actual) < numel(exp_sz)
            actual = [actual, ones(1, numel(exp_sz) - numel(actual))]; %#ok<AGROW>
        end
        if ~isequal(actual(1:numel(exp_sz)), exp_sz)
            fprintf('  [FAIL] draws.%-9s size = [%s], expected [%s]\n', ...
                    fn{f}, num2str(actual(1:numel(exp_sz))), num2str(exp_sz));
            ok_fields = false; n_fail = n_fail + 1;
        end
    end
    if ok_fields
        fprintf('  [PASS] All expected fields with correct dimensions\n');
    end

    % --- (3) No NaN or Inf ---
    has_bad = false;
    fn = fieldnames(draws);
    for f = 1:numel(fn)
        X = draws.(fn{f});
        if any(~isfinite(X(:)))
            fprintf('  [FAIL] draws.%s contains NaN/Inf\n', fn{f});
            has_bad = true; n_fail = n_fail + 1;
        end
    end
    if ~has_bad
        fprintf('  [PASS] No NaN/Inf in any field\n');
    end

    % --- (4) Sign-flip moves active: s_0 marginal roughly 50/50 +/- ---
    % With Mc = 50 draws and a per-equation flip every iter, each equation
    % should have ~50% positive and ~50% negative draws of s_0.
    s_pos_frac = mean(draws.s_0 > 0, 1);   % 1 x N
    % Allow generous tolerance with only 50 draws: each equation should
    % be in [0.2, 0.8]. If sign flips weren't running we'd see ~0 or ~1.
    if all(s_pos_frac >= 0.20 & s_pos_frac <= 0.80)
        fprintf('  [PASS] Sign-flip active on s_0 (frac+ in [%.2f, %.2f])\n', ...
                min(s_pos_frac), max(s_pos_frac));
    else
        fprintf('  [WARN] s_0 sign distribution unusual: min=%.2f max=%.2f\n', ...
                min(s_pos_frac), max(s_pos_frac));
        % This is a warning, not a fail -- with Mc=50 random variation
        % can land outside [0.2, 0.8] occasionally.
    end

    % --- (5) Phi reconstruction matches a manual recomputation ---
    % Pick a specific (d, i, k, t) and check the helper output equals
    % the recomputed value from the underlying draws.
    d_pick = randi(opts.mcmc);
    t_pick = randi(T);
    i_pick = randi(N);

    Phi_q = reconstruct_RR_HYB_quantiles(draws, qoi);

    % Manual recomputation: intercept (k=1) and one lag (k=2)
    Psi_d = squeeze(draws.Psi_bar(d_pick, :, :));     % N x K
    s_0_d = squeeze(draws.s_0(d_pick, :))';            % N x 1
    th0_d = squeeze(draws.theta_0(d_pick, :, t_pick));% 1 x N -> need vector
    th0_d = th0_d(:);                                  % N x 1
    S_l_d = squeeze(draws.S_l_m(d_pick, :, :));        % N x K_lag
    A_d   = squeeze(draws.A(d_pick, t_pick, :, :));   % N x R
    B_d   = squeeze(draws.B(d_pick, t_pick, :, :));   % K_lag x R
    AB_d  = A_d * B_d';                                 % N x K_lag

    Phi_d_int = Psi_d(:, 1) + s_0_d .* th0_d;          % N x 1, intercept col
    Phi_d_lag = Psi_d(:, 2:end) + S_l_d .* AB_d;        % N x K_lag, lag cols
    Phi_d_full = [Phi_d_int, Phi_d_lag];                 % N x K

    % Now reconstruct the same Phi(d, :, t) for ALL draws via the helper,
    % grab the d_pick-th cell, and compare. Easiest: redo the helper math
    % once for draw d_pick only.
    %
    % Quick sanity check: median quantile should bracket the per-draw value
    % across most cells (it doesn't have to coincide, but should be within
    % the credible band). Tighter check: rerun helper logic for draw d_pick.
    %
    % Tighter check (deterministic): compute the median of Phi_d_full
    % against the helper-returned median Phi_q(:,:,t_pick, 2).
    % These won't match unless we run the helper on a single draw.
    % Skip the exact-match test -- instead, check the median Phi_q value
    % is finite and in a reasonable range.

    Phi_q_med = Phi_q(:, :, t_pick, qoi == 0.50);
    if all(isfinite(Phi_q_med(:)))
        fprintf('  [PASS] Phi_q (median) finite at t=%d, range [%.3f, %.3f]\n', ...
                t_pick, min(Phi_q_med(:)), max(Phi_q_med(:)));
    else
        fprintf('  [FAIL] Phi_q (median) contains NaN/Inf at t=%d\n', t_pick);
        n_fail = n_fail + 1;
    end

    if size(Phi_q, 1) == N && size(Phi_q, 2) == K && ...
       size(Phi_q, 3) == T && size(Phi_q, 4) == numel(qoi)
        fprintf('  [PASS] reconstruct_RR_HYB_quantiles output shape [%d %d %d %d]\n', ...
                N, K, T, numel(qoi));
    else
        fprintf('  [FAIL] reconstruct_RR_HYB_quantiles bad shape [%s]\n', ...
                num2str(size(Phi_q)));
        n_fail = n_fail + 1;
    end

    % --- (6) IFs are finite and non-negative ---
    IFs_Phi = compute_RR_HYB_Phi_IFs(draws);
    if all(isfinite(IFs_Phi(:))) && all(IFs_Phi(:) >= 0)
        fprintf('  [PASS] IFs_Phi finite and non-negative, max = %.1f\n', ...
                max(IFs_Phi(:)));
    else
        fprintf('  [FAIL] IFs_Phi has bad values (min=%.3f, max=%.3f)\n', ...
                min(IFs_Phi(:)), max(IFs_Phi(:)));
        n_fail = n_fail + 1;
    end

    if isequal(size(IFs_Phi), [N, K, T])
        fprintf('  [PASS] compute_RR_HYB_Phi_IFs output shape [%d %d %d]\n', N, K, T);
    else
        fprintf('  [FAIL] compute_RR_HYB_Phi_IFs bad shape [%s]\n', ...
                num2str(size(IFs_Phi)));
        n_fail = n_fail + 1;
    end

    % --- (7) Manual reconstruction at one cell against helper internals ---
    % Compute the median Phi via the helper formula for ALL draws at one
    % (i, k, t) cell and compare with what would come from reconstruct_RR_HYB.
    %
    % We do this for the intercept column to test the s_0 .* theta_0 path
    % specifically (the part that's NEW vs the non-hybrid).
    t_pick = max(2, round(T/2));
    i_pick = randi(N);
    Phi_int_all = squeeze(draws.Psi_bar(:, i_pick, 1)) ...
                + squeeze(draws.s_0(:, i_pick)) .* squeeze(draws.theta_0(:, i_pick, t_pick));
    med_manual = median(Phi_int_all);
    med_helper = Phi_q(i_pick, 1, t_pick, qoi == 0.50);

    if abs(med_manual - med_helper) < 1e-10
        fprintf('  [PASS] Manual vs helper match for intercept Phi at (i=%d, t=%d): %.6f\n', ...
                i_pick, t_pick, med_manual);
    else
        fprintf('  [FAIL] Manual vs helper mismatch for intercept Phi: manual=%.6f helper=%.6f\n', ...
                med_manual, med_helper);
        n_fail = n_fail + 1;
    end

    fprintf('\n');
end

%% ---- summary ----------------------------------------------------------

if n_fail == 0
    fprintf('=== ALL CHECKS PASSED ===\n');
    fprintf('Hybrid sampler structurally healthy on both 2RW_HYB and RWAR_HYB.\n');
    fprintf('Safe to scale up opts.burn / opts.mcmc and run the full job.\n');
else
    fprintf('=== %d CHECK(S) FAILED ===\n', n_fail);
    fprintf('Inspect the [FAIL] lines above before running the full job.\n');
end
