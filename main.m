%% main.m
% Driver for the Reduced Rank Time Variation paper.
%
% Choose dataset (small / medium / large), choose time span, optionally
% plot. Estimation hooks will be wired in later as the sampler is built.

clear; 
clc; 
close all;

rng(2026);

addpath(genpath(fullfile(pwd, 'Functions')));
addpath(genpath(fullfile(pwd, 'Models')));

%% ----- General settings -------------------------------------------------

dataset    = 'small';               % 'small' | 'medium' | 'large'

date_start = datetime(1959, 6, 1);  % inclusive; first available 1959-Q2
date_end   = datetime(2019, 12, 1); % inclusive; pre Covid

do_plot    = false;

%% ----- VAR settings -----------------------------------------------------

P = 2;  % VAR lag order
R = 1;  % rank of the decomposition

%% ----- MCMC settings ----------------------------------------------------

opts.burn       = 2e3;
opts.mcmc       = 1e4;
opts.thin       = 1;
opts.print_freq = 1e3;

%% ----- Load and slice by time span --------------------------------------

[Y, Names, Ydates, T, N] = load_slice(dataset, date_start, date_end);

if do_plot
    plot_dataset(Y, Ydates, Names); 
end

%% ----- Priors -----------------------------------------------------------

priors.lam    = [10^2; 0.2^2; 0.1^2; 2]; % [const, own, cross, decay]
priors.sh_lam = 2;                       % Gamma shape

priors.V_sl = 0.01^2; % prior var of each element of diag(S) (Chan-Strachan: intercept + lags share one loading)

priors.Sigma_nu_offset = 2; % nu = N + 2 

%% ----- Estimation ------------------------------------------------------

draws = MCMC_RR_TVPVAR(Y, P, R, priors, opts);
