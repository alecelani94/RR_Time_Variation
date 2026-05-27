clear; 
clc; 
close all;

rng(2026);

addpath(genpath(fullfile(pwd, 'Functions')));
addpath(genpath(fullfile(pwd, 'Models')));

%% ----- General settings -------------------------------------------------

dataset    = 'small_3';             % 'small_3' | 'small_5' | 'medium_7' | 'medium_11' | 'large'

date_start = datetime(1959, 6, 1);  % inclusive; first available 1959-Q2
date_end   = datetime(2019, 12, 1); % inclusive; pre Covid

do_plot    = true;

%% ----- VAR settings -----------------------------------------------------

P = 2;  % VAR lag order
% R is swept: ranks 1..R_max are estimated, where R_max is the smallest R
% such that the bilinear spec has at least as many TV parameters as the
% unrestricted spec. R_max is computed from (N, P) after the data is loaded.

% RR law-of-motion variant:
%   '2RW'   - both A and B follow RW (symmetric; quadratic variance growth)
%   'RWAR'  - one factor RW, the other AR(1) (linear variance growth)
rr_variant = 'RWAR';     % '2RW' | 'RWAR'
ar_factor  = 'A';        % 'A' | 'B'  -- which factor is AR(1) (used iff rr_variant = 'RWAR')
rho_ar     = 0.99;       % AR(1) persistence (used iff rr_variant = 'RWAR')

%% ----- MCMC settings ----------------------------------------------------

opts.burn       = 5e3;
opts.mcmc       = 1e4;
opts.thin       = 1;
opts.print_freq = 5e3;

%% ----- Posterior summaries ----------------------------------------------

levels = [.95, .90, .75, .68, .50];
qoi    = sort(unique([(1 - levels)/2, 1 - (1 - levels)/2, 0.50]));

%% ----- Load and slice by time span --------------------------------------

[Y, Names, Ydates, T, N] = load_slice(dataset, date_start, date_end);

if do_plot
    plot_dataset(Y, Ydates, Names); %#ok<UNRCH>
end

%% ----- Priors -----------------------------------------------------------

% --- Static block (Minnesota) ---
priors.lam      = [10^2; 0.2^2; 0.1^2; 2];   % [const, own, cross, decay]
priors.sh_lam   = 2;                         % Gamma shape on lambda_own/cross
priors.lam_hier = false;                      % sample lambda_own/cross via GIG

% --- TV loading (diag of S) ---
priors.V_sl_intercept = 0.05^2;              % prior var of TV INTERCEPT loading (S col 1)
priors.V_sl_lag       = 0.01^2;              % prior var of TV LAG loading       (S cols 2..K)

% --- Residual covariance ---
priors.Sigma_nu_offset = 2;                  % IW dof: nu = N + offset

%% ----- Rank-sweep range -------------------------------------------------
% Unrestricted TV parameter count per period: M = N*K = N*(N*P+1).
% Rank-R bilinear TV parameter count per period: (N + K)*R.
% R_max = largest R such that the bilinear count is STRICTLY less than the
% unrestricted count: (N+K)*R_max < M  <=>  R_max = floor((M-1)/(N+K)).
% At R_max+1 the bilinear would meet or exceed the unrestricted spec.

K_var  = N * P + 1;
M_full = N * K_var;
R_max  = floor((M_full - 1) / (N + K_var));
fprintf('Rank sweep: R = 1..%d (unrestricted M = %d, N+K = %d)\n', ...
        R_max, M_full, N + K_var);

%% ----- Estimation: unrestricted TVP benchmark --------------------------

fprintf('\n=== Unrestricted TVP-VAR ===\n');
draws_TVP   = MCMC_TVPVAR(Y, P, priors, opts);

IFs_TVP     = compute_IFs(draws_TVP);
IFs_TVP.Phi = compute_TVP_Phi_IFs(draws_TVP);
Phi_q_TVP   = reconstruct_TVP_quantiles(draws_TVP, qoi);
draws_TVP   = rmfield(draws_TVP, 'theta');

%% ----- Estimation: rank-R RR for R = 1..R_max --------------------------

draws_RR = cell(R_max, 1);
IFs_RR   = cell(R_max, 1);
Phi_q_RR = cell(R_max, 1);

for R = 1:R_max

    fprintf('\n=== RR bilinear TVP-VAR (%s, R = %d / %d) ===\n', rr_variant, R, R_max);
    switch upper(rr_variant)
        case '2RW'
            draws_r = MCMC_RR_2RW_TVPVAR(Y, P, R, priors, opts);
        case 'RWAR'
            priors.ar_factor = ar_factor;
            priors.rho       = rho_ar;
            draws_r = MCMC_RR_RWAR_TVPVAR(Y, P, R, priors, opts);
        otherwise
            error('main_fs:badVariant', ...
                  'rr_variant must be ''2RW'' or ''RWAR'' (got ''%s'').', rr_variant);
    end
    IFs_r         = compute_IFs(draws_r);
    IFs_r.Phi     = compute_RR_Phi_IFs(draws_r);
    Phi_q_r       = reconstruct_RR_quantiles(draws_r, qoi);
    draws_r       = rmfield(draws_r, {'A', 'B'});

    draws_RR{R}   = draws_r;
    IFs_RR{R}     = IFs_r;
    Phi_q_RR{R}   = Phi_q_r;
end

%% ----- Save -------------------------------------------------------------

if ~exist('Output', 'dir'); mkdir('Output'); end
% Filename includes the RR variant tag so 2RW and RWAR runs don't overwrite.
out_path = fullfile(pwd, 'Output', ...
    sprintf('%s_P%d_%s.mat', dataset, P, rr_variant));
save(out_path, ...
    'draws_TVP',  'IFs_TVP',  'Phi_q_TVP', ...
    'draws_RR',   'IFs_RR',   'Phi_q_RR',  'R_max', ...
    'qoi',        'levels', ...
    'priors',     'opts',     'P', ...
    'rr_variant', 'ar_factor', 'rho_ar', ...
    'dataset',    'date_start', 'date_end', ...
    'Y',         'Names',     'Ydates');
fprintf('\nSaved to: %s\n', out_path);
