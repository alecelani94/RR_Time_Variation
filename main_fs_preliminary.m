%% main_fs_preliminary.m -- Posterior median of the TV coefficient matrix
%
% Estimates the unrestricted TVP-VAR benchmark and saves the posterior
% median of the (identified) TV coefficient matrix tensor
%
%     Theta_t^{(m)}  = reshape(s^{(m)} .* theta_t^{(m)}, N, K),
%     Theta_med      = median_m Theta_t^{(m)}    ->  N x K x T.
%
% Theta_t is sign-invariant (the (s, theta) sign ambiguity cancels in the
% product), so its posterior is unimodal and the median is a clean point
% summary. All rank-diagnostic decompositions (full N x K SVD, lag-only
% N x N*P SVD, per-lag N x N SVDs, alternative normalizers, etc.) are
% deferred to eval_fs_preliminary.m so the choice of decomposition can be
% changed without rerunning MCMC.

clear;
clc;
close all;

rng(2026);

addpath(genpath(fullfile(pwd, 'Functions')));
addpath(genpath(fullfile(pwd, 'Models')));

%% ----- General settings -------------------------------------------------

dataset    = 'medium_11';

date_start = datetime(1959, 6, 1);
date_end   = datetime(2019, 12, 1);

%% ----- VAR settings -----------------------------------------------------

P = 4;

%% ----- MCMC settings ----------------------------------------------------

opts.burn       = 5e3;
opts.mcmc       = 1e4;
opts.thin       = 1;
opts.print_freq = 5e2;

%% ----- Load and slice by time span --------------------------------------

[Y, Names, Ydates, T, N] = load_slice(dataset, date_start, date_end);

%% ----- Priors (matched to main_fs.m) -----------------------------------

priors.lam      = [10^2; 0.2^2; 0.1^2; 2];
priors.sh_lam   = 2;
priors.lam_hier = false;

priors.V_sl_intercept = 0.05^2;
priors.V_sl_lag       = 0.01^2;

priors.Sigma_nu_offset = 2;

%% ----- Sizes -------------------------------------------------------------

K_var  = N * P + 1;
M_full = N * K_var;
K_lag  = N * P;
fprintf('N = %d, P = %d, K = %d, M = %d\n', N, P, K_var, M_full);

%% ----- Estimation: unrestricted TVP-VAR --------------------------------

fprintf('\n=== Unrestricted TVP-VAR ===\n');
draws_TVP = MCMC_TVPVAR(Y, P, priors, opts);

%% ----- Posterior median of Theta_t = reshape(s .* theta) ----------------
%
% draws_TVP.s     : Mc x M
% draws_TVP.theta : Mc x M x T
% Broadcast: s (Mc x M) .* theta (Mc x M x T) -> Mc x M x T (implicit
% singleton on s's third dim, R2016b+). Median across draws -> M x T,
% reshape -> N x K x T.

Mc    = size(draws_TVP.theta, 1);
T_use = size(draws_TVP.theta, 3);

fprintf('Computing posterior median Theta_t (Mc = %d, T = %d) ...\n', Mc, T_use);
t_tic = tic;
S_theta   = draws_TVP.s .* draws_TVP.theta;
Theta_med = reshape(squeeze(median(S_theta, 1)), N, K_var, T_use);
fprintf('Done in %.1fs.\n', toc(t_tic));

%% ----- Save -------------------------------------------------------------

if ~exist('Output', 'dir'); mkdir('Output'); end
out_path = fullfile(pwd, 'Output', sprintf('%s_P%d_preliminary.mat', dataset, P));
save(out_path, ...
    'Theta_med', ...
    'Mc', 'T_use', 'N', 'K_var', 'K_lag', ...
    'priors', 'opts', 'P', 'dataset', 'date_start', 'date_end', ...
    'Y', 'Names', 'Ydates');
fprintf('\nSaved to: %s\n', out_path);
