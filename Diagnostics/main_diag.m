% MAIN_DIAG  Rank-budget pre-check for the hybrid RR-TVP spec.
%
% Runs the unrestricted TVP-VAR sampler on the chosen dataset and computes
% two diagnostics on the resulting draws of the TV coefficient matrix
% (see brainstorm doc Sec.3, D1 and D2):
%
%   D1: intercept share of the TV signal (overall and per equation).
%   D2: singular value decay of the N x NP lag block.
%
% These confirm (or invalidate) the rank-budget story that motivates the
% hybrid spec BEFORE any new sampler is implemented.
%
% Outputs are saved to ../Output/Diagnostics/. Plots are NOT generated here
% (server-friendly); run plot_rank_budget_diag.m locally on the saved .mat.

clear;
clc;
close all;

rng(2026);

%% ---- path setup -------------------------------------------------------

this_dir   = fileparts(mfilename('fullpath'));
parent_dir = fileparts(this_dir);

addpath(genpath(fullfile(parent_dir, 'Functions')));
addpath(genpath(fullfile(parent_dir, 'Models')));
addpath(this_dir);

%% ---- settings (mirror main_fs.m so the posterior is the same) ---------

dataset    = 'small_5_gs5';
date_start = datetime(1959, 6, 1);
date_end   = datetime(2019, 12, 1);

P = 2;

opts.burn       = 5e3;
opts.mcmc       = 1e4;
opts.thin       = 1;
opts.print_freq = 5e3;

%% ---- priors (mirror main_fs.m) ----------------------------------------

priors.lam      = [10^2; 0.2^2; 0.1^2; 2];
priors.sh_lam   = 2;
priors.lam_hier = false;

priors.V_sl_intercept = 0.05^2;
priors.V_sl_lag       = 0.01^2;

priors.Sigma_nu_offset = 2;

%% ---- load data --------------------------------------------------------

[Y, Names, Ydates, T, N] = load_slice(dataset, date_start, date_end);
fprintf('Loaded %s: N=%d, T=%d, P=%d\n', dataset, N, T, P);

%% ---- run unrestricted TVP-VAR ----------------------------------------

fprintf('\n=== Unrestricted TVP-VAR (for rank-budget diagnostics) ===\n');
t_sampler = tic;
draws_TVP = MCMC_TVPVAR(Y, P, priors, opts);
fprintf('Sampler done in %.1f s\n', toc(t_sampler));

%% ---- compute D1 + D2 --------------------------------------------------

fprintf('\n=== Rank-budget diagnostics (D1 + D2) ===\n');
t_diag = tic;
summary = diagnostics_TVP_rank_budget(draws_TVP, N, P);
fprintf('Diagnostics done in %.1f s\n', toc(t_diag));

%% ---- one-line CLI snapshot -------------------------------------------
% Print three numbers to stdout so the server log immediately tells the
% story without needing the .mat to be downloaded.

med_rho_int = median(summary.rho_int_q(:, summary.q == 0.5));
fprintf('\n[Snapshot] Median over t of posterior-median rho_int (D1): %.3f\n', med_rho_int);

if isfield(summary.topR_share, 'R3')
    med_top3 = median(summary.topR_share.R3(:, summary.q == 0.5));
    fprintf('[Snapshot] Median over t of posterior-median top-3 SV share (D2): %.3f\n', med_top3);
end

% Worst row (highest median rho_int_row averaged over t)
rho_int_row_med = squeeze(summary.rho_int_row_q(:, :, summary.q == 0.5));   % N x T
[worst_med, worst_idx] = max(mean(rho_int_row_med, 2));
fprintf('[Snapshot] Worst-row equation: %s (mean median rho_int = %.3f)\n', ...
        Names{worst_idx}, worst_med);

%% ---- save -------------------------------------------------------------

out_dir = fullfile(parent_dir, 'Output', 'Diagnostics');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
out_path = fullfile(out_dir, sprintf('rank_budget_%s_P%d.mat', dataset, P));

save(out_path, ...
    'summary', ...
    'priors', 'opts', 'P', ...
    'dataset', 'date_start', 'date_end', ...
    'N', 'T', 'Names', 'Ydates');
fprintf('\nSaved diagnostics to: %s\n', out_path);
