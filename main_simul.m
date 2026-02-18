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

b = 0.20;  % elemenst of the B matrix (same for all)

% mcmc
burn       = 2e3;
mcmc       = 1e4;
thin       = 1;
print_freq = 2.5e3;

%% ----- Static VAR coefficients (Phi_bar) --------------------------------

Phi_bar = [ .62  -.29   .23;
            .15   .54  -.37;
           -.31   .08   .70];

max_eig = max(abs(eig(Phi_bar)));
if max_eig > 1

   error('Unstable system');
end

%% ----- Static error covariance matrix (Sigma) ---------------------------

Sigma = eye(N);
% Sigma = [ 1   .6  .4;
%           .6   1  .5;
%           .4  .5  1.0];

try
    Sigma_chol = chol(Sigma, 'lower');
catch
    error('Sigma is not positive definite.');
end

%% ----- Deterministic sine paths for a(t) and c(t) -----------------------

% a_i(t) = amplitude * sin(2*pi*freq_a(i) * (t-1)/T)
% c_j(t) = amplitude * sin(2*pi*freq_c(j) * (t-1)/T)
% All paths: initialized at 0, span [-0.99, 0.99], max 2 peaks.

amplitude = 1;
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

% wrap true parameters for initial conditions
init.Phi_bar = Phi_bar;
init.B       = B;
init.a       = a;
init.c       = c;
init.sigma2  = diag(Sigma);

%% ----- Plots ------------------------------------------------------------

% plot_simul;

%% ----- Estimate ---------------------------------------------------------

% prior variances mean coeffs
lambda = [.5^2;    % static                   : Phi_bar(i,j)
          .2^2];  % st. dev. dynamic decomp. : B(i,j) 

% IG prior individual variances
gamma = [4;   % shape
         1];  % mean


pos_rw  = 1;
ar1_par = .99;

get_par_aux;

[Phi_bar_draws, B_draws, a_draws, c_draws, sigma2_draws] = ...
 MCMC_bilinear_TVP_VAR1_simul                              ...
(Y, lambda, gamma, par_aux, init,                ...
  burn, mcmc, thin, print_freq);

%% ----- Static coefficients: trace plots ---------------------------------

figure('Name','Phi_bar draws');
tiledlayout(N, N, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:N
    for j = 1:N
        nexttile;
        plot(squeeze(Phi_bar_draws(:,i,j)), 'r');
        yline(Phi_bar(i,j), '-.k', 'LineWidth', 2);
        title(sprintf('\\Phi_{%d,%d}', i, j));
    end
end

figure('Name','B draws');
tiledlayout(N, N, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:N
    for j = 1:N
        nexttile;
        plot(squeeze(B_draws(:,i,j)), 'r');
        % yline(B(i,j), '-.k', 'LineWidth', 2);
        title(sprintf('B_{%d,%d}', i, j));
    end
end

figure('Name','sigma2 draws');
tiledlayout(N, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:N
    
    nexttile;
    plot(squeeze(sigma2_draws(:,i)), 'r');
    yline(Sigma(i,i), '-.k', 'LineWidth', 2);
    title(sprintf('sigma2_{%d}', i));
end

%% ----- a(t) and c(t): median and 68% bands ------------------------------

a_med = squeeze(median(a_draws, 1));   % T x N
a_lo  = squeeze(quantile(a_draws, 0.16, 1));
a_hi  = squeeze(quantile(a_draws, 0.84, 1));

figure('Name','a(t) paths');
tiledlayout(N, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:N
    nexttile; hold on;
    fill([1:T, T:-1:1], [a_lo(:,i)', fliplr(a_hi(:,i)')], ...
         [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    plot(a_med(:,i), 'r', 'LineWidth', 1.5);
    plot(a(:,i), '--k', 'LineWidth', 1.5);
    hold off;
    title(sprintf('a_{%d}(t)', i));
end

c_med = squeeze(median(c_draws, 1));   % T x N
c_lo  = squeeze(quantile(c_draws, 0.16, 1));
c_hi  = squeeze(quantile(c_draws, 0.84, 1));

figure('Name','c(t) paths');
tiledlayout(N, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:N
    nexttile; hold on;
    fill([1:T, T:-1:1], [c_lo(:,i)', fliplr(c_hi(:,i)')], ...
         [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    plot(c_med(:,i), 'r', 'LineWidth', 1.5);
    plot(c(:,i), '--k', 'LineWidth', 1.5);
    hold off;
    title(sprintf('c_{%d}(t)', i));
end

%% ----- TV coefficients: median and 68% bands ----------------------------

Phi_tv_med = zeros(N, N, T);
Phi_tv_lo  = zeros(N, N, T);
Phi_tv_hi  = zeros(N, N, T);

for i = 1:N
    for j = 1:N
        % mcmc x T: Phi_bar(i,j) + a_i(t) * B(i,j) * c_j(t)
        phi_ij = Phi_bar_draws(:,i,j) + a_draws(:,:,i) .* B_draws(:,i,j) .* c_draws(:,:,j);
        Phi_tv_med(i,j,:) = median(phi_ij, 1);
        Phi_tv_lo(i,j,:)  = quantile(phi_ij, 0.16, 1);
        Phi_tv_hi(i,j,:)  = quantile(phi_ij, 0.84, 1);
    end
end

figure('Name','TV coefficients');
tiledlayout(N, N, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:N
    for j = 1:N
        nexttile;
        hold on;
        fill([1:T, T:-1:1], ...
             [squeeze(Phi_tv_lo(i,j,:))', fliplr(squeeze(Phi_tv_hi(i,j,:))')], ...
             [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        plot(squeeze(Phi_tv_med(i,j,:)), 'r', 'LineWidth', 1.5);
        plot(squeeze(Phi_tv(i,j,:)), '--k', 'LineWidth', 1.5);
        hold off;
        title(sprintf('\\Phi_{%d,%d}(t)', i, j));
    end
end

%% ----- ESS --------------------------------------------------------------

% % Phi_bar
% ESS_Phi = zeros(N, N);
% for i = 1:N
%     for j = 1:N
%         ESS_Phi(i,j) = ess(squeeze(Phi_bar_draws(:,i,j)));
%     end
% end
% 
% % B
% ESS_B = zeros(N, N);
% for i = 1:N
%     for j = 1:N
%         ESS_B(i,j) = ess(squeeze(B_draws(:,i,j)));
%     end
% end
% 
% % sigma2
% ESS_sigma2 = zeros(N, 1);
% for i = 1:N
%     ESS_sigma2(i) = ess(sigma2_draws(:,i));
% end
% 
% % a(t), c(t): ESS per variable (average across time)
% ESS_a = zeros(T, N);
% ESS_c = zeros(T, N);
% for i = 1:N
%     for t = 1:T
%         ESS_a(t,i) = ess(a_draws(:,t,i));
%         ESS_c(t,i) = ess(c_draws(:,t,i));
%     end
% end
% 
% % boxplot
% n_cols = 4;
% boxplot_data = cell(1, n_cols);
% boxplot_data{1} = ESS_Phi(:);
% boxplot_data{2} = ESS_B(:);
% boxplot_data{3} = ESS_sigma2;
% boxplot_data{4} = [ESS_a(:); ESS_c(:)];
% 
% boxplot_labels = {'\Phi', 'B', '\sigma^2', 'a, c'};
% 
% max_len = max(cellfun(@length, boxplot_data));
% boxplot_matrix = NaN(max_len, n_cols);
% for k = 1:n_cols
%     boxplot_matrix(1:length(boxplot_data{k}), k) = boxplot_data{k};
% end
% 
% figure('Name','ESS by parameter class');
% h = boxplot(boxplot_matrix, 'Labels', boxplot_labels, 'Notch', 'on');
% set(h, 'LineWidth', 1.5);
% title('ESS by parameter class');
