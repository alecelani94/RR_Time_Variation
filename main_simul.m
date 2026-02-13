%% main_simul.m
% =========================================================================
% Simulate data from a Bilinear TVP-VAR(1) model
%
% Model (non-centered parameterization):
%   y_t = Phi_bar * y_{t-1} + Phi_tilde_t * y_{t-1} + eps_t
%
% Bilinear decomposition:
%   Phi_tilde_t = A_t * B * C_t
%
% where A_t = diag(a_t), C_t = diag(c_t), and:
%   a_t, c_t: deterministic sine functions

clear; 
clc; 
close all;

rng(2026);

addpath(genpath(pwd));

%% ----- Settings ---------------------------------------------------------

N = 3;    
T = 500;  
P = 1; 

b = 0.05;  % elemenst of the B matrix (same for all)

% MCMC
Burn = 2e3;
MCMC = 1e4;
Thin = 1;

%% ----- Static VAR coefficients (Phi_bar) --------------------------------

Phi_bar = [ .70  -.34   .26;
            .17   .61  -.42;
           -.36   .10   .79];

max_eig = max(abs(eig(Phi_bar)));
if max_eig > 1

   error('Unstable system');
end

%% ----- Static error covariance matrix (Sigma) ---------------------------

Sigma = [ 1   .6  .4;
          .6   1  .5;
          .4  .5  1.0];
try
    Sigma_chol = chol(Sigma, 'lower');
catch
    error('Sigma is not positive definite.');
end

%% ----- Deterministic sine paths for a(t) and c(t) -----------------------

% a_i(t) = amplitude * sin(2*pi*freq_a(i) * (t-1)/T)
% c_j(t) = amplitude * sin(2*pi*freq_c(j) * (t-1)/T)
% All paths: initialized at 0, span [-0.99, 0.99], max 2 peaks.

amplitude = 0.95;
freq_a = [1.25; 1.0; 0.75];  % cycles per sample for a_1, a_2, a_3
freq_c = [0.75; 1.0; 1.25];  % cycles per sample for c_1, c_2, c_3

a = zeros(T, N);
c = zeros(T, N);

time_grid = ((1:T) - 1) / T;   % normalized: 0, 1/T, ..., (T-1)/T

for i = 1:N
    a(:,i) = amplitude * sin(2*pi*freq_a(i) * time_grid);
    c(:,i) = amplitude * sin(2*pi*freq_c(i) * time_grid);
end
a(:,2)     = -a(:,2);
c(:,[1,3]) = -c(:,[1,3]);

%% ----- Construct TV matrices and simulate -------------------------------

% Phi_tilde_t = diag(a_t) * B * diag(c_t)
% Phi_t = Phi_bar + Phi_tilde_t

B         = b * ones(N, N);
Phi_tilde = zeros(N, N, T);
Phi_tv    = zeros(N, N, T);

max_eigt  = zeros(T, 1);
Y         = zeros(N, T);

for t = 1:T

    A_t = diag(a(t,:));
    C_t = diag(c(t,:));

    Phi_tilde(:,:,t) = A_t * B * C_t;

    Phi_tv(:,:,t) = Phi_bar + Phi_tilde(:,:,t);

    max_eigt(t) = max(abs(eig(Phi_tv(:,:,t))));

    if t == 1

       Y(:,t) = Sigma_chol * randn(N, 1);
    else

       Y(:, t) = Phi_tv(:,:,t) * Y(:, t-1) + Sigma_chol * randn(N, 1);
    end
end
Y = Y';

if any(max_eigt > 1)
    
   error('Unstable system');
end

%% ----- Plots ------------------------------------------------------------

plot_simul;

%% ----- Estimate ---------------------------------------------------------


% Minnesota prior: [constant, own lag, cross lag, contemporaneous, lag decay]
lambda = [10^2, .2^2, .1^2, .2^2, 2];

% results = MCMC_bilinear_TVP_VAR1_simul(Y, lambda, Burn, MCMC, Thin);


