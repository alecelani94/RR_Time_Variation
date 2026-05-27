%% eval_fs_preliminary.m -- Cross-dataset rank decomposition (median)
%
% Loads Output/<dataset>_P<P>_preliminary.mat for each dataset in the
% list, computes the FVE curves on the LAG block of Theta_med, and plots
% a 1 x 2 figure:
%
%   Panel (ii):  bilinear on lag-only block (best rank-R for the lag block).
%   Panel (iii): per-lag bilinear, summed across lags.
%
% Each panel shows one curve per dataset (color-coded by N), same x-axis
% (rank R) for direct visual comparison across system sizes. Curves end at
% their own N (the matrix rank cap).

clear;
clc;
close all;

addpath(genpath(fullfile(pwd, 'Functions')));

%% ---- Selections -------------------------------------------------------

P_sel = 4;

%% ---- Auto-discover preliminary outputs in Output/ ---------------------

flist = dir(fullfile(pwd, 'Output', sprintf('*_P%d_preliminary.mat', P_sel)));
if isempty(flist)
    error('eval_fs_preliminary:noFiles', ...
          'No *_P%d_preliminary.mat found in Output/.', P_sel);
end

suffix   = sprintf('_P%d_preliminary.mat', P_sel);
datasets = cell(1, numel(flist));
for d = 1:numel(flist)
    datasets{d} = erase(flist(d).name, suffix);
end

%% ---- Loop over datasets, compute (ii) and (iii) FVE curves ------------

nD = numel(datasets);
fve_ii_all  = cell(nD, 1);
fve_iii_all = cell(nD, 1);
N_all       = nan(nD, 1);
labels      = cell(nD, 1);

for d = 1:nD
    ds   = datasets{d};
    fpth = fullfile(pwd, 'Output', sprintf('%s_P%d_preliminary.mat', ds, P_sel));

    S = load(fpth);
    [Ndim, ~, Tdim] = size(S.Theta_med);
    N_d = Ndim;
    P_d = S.P;

    fve_ii_t  = zeros(Tdim, N_d);
    fve_iii_t = zeros(Tdim, N_d);

    for t = 1:Tdim
        Theta_t   = S.Theta_med(:, :, t);
        lag_block = Theta_t(:, 2:end);
        denom     = max(sum(lag_block(:).^2), eps);

        % --- (ii) lag-only SVD ---
        sv2_lag = svd(lag_block).^2;
        fve_ii_t(t, :) = cumsum(sv2_lag(:)).' / denom;

        % --- (iii) per-lag SVDs, summed across lags ---
        per_lag_cum = zeros(N_d, 1);
        for p = 1:P_d
            cols = (p-1)*N_d + 2 : p*N_d + 1;
            per_lag_cum = per_lag_cum + cumsum(svd(Theta_t(:, cols)).^2);
        end
        fve_iii_t(t, :) = per_lag_cum.' / denom;
    end

    fve_ii_all{d}  = mean(fve_ii_t,  1);
    fve_iii_all{d} = mean(fve_iii_t, 1);
    N_all(d)       = N_d;
    labels{d}      = sprintf('N=%d', N_d);
end

% Sort by N (ascending) so legend reads from smallest to largest system.
[~, ord]   = sort(N_all);
idx        = ord(:)';
fprintf('Found %d preliminary file(s) for P=%d:\n', nD, P_sel);
for k = 1:numel(idx)
    fprintf('  %s\n', labels{idx(k)});
end

ds_color = lines(nD);
ds_style = repmat({'-'}, 1, nD);

%% ---- Plot 1 x 2: panel per method, line per dataset -------------------

max_R = max(N_all(idx));

figure('Name', sprintf('Rank diagnostic across datasets (P=%d)', P_sel), ...
       'Position', [50 50 1300 480]);

y_lim = [0.475 1.025];

% --- Panel (ii) ---
subplot(1, 2, 1);
hold on;
h_ii = gobjects(1, numel(idx));
for k = 1:numel(idx)
    d = idx(k);
    h_ii(k) = plot(1:N_all(d), fve_ii_all{d}, ...
                   '-o', 'LineWidth', 1.6, ...
                   'Color', ds_color(k, :), 'MarkerFaceColor', ds_color(k, :));
end
xlabel('Rank R');
ylabel('FVE');
title('Lag-only');
xlim([0.5, max_R + 0.5]);
ylim(y_lim);
grid on;
legend(h_ii, labels(idx), 'Location', 'southeast', ...
       'FontSize', 11, 'Interpreter', 'none');

% --- Panel (iii) ---
subplot(1, 2, 2);
hold on;
h_iii = gobjects(1, numel(idx));
for k = 1:numel(idx)
    d = idx(k);
    h_iii(k) = plot(1:N_all(d), fve_iii_all{d}, ...
                    '-o', 'LineWidth', 1.6, ...
                    'Color', ds_color(k, :), 'MarkerFaceColor', ds_color(k, :));
end
xlabel('Rank R');
ylabel('FVE');
title('Per-lag, summed');
xlim([0.5, max_R + 0.5]);
ylim(y_lim);
grid on;

%% ---- Save the figure to Overleaf/figures folder -----------------------

fig_dir = fullfile(pwd, 'Overleaf', 'figures');
if ~exist(fig_dir, 'dir'); mkdir(fig_dir); end
out_pdf = fullfile(fig_dir, 'rank_diagnostic.pdf');
exportgraphics(gcf, out_pdf, 'ContentType', 'vector');
fprintf('Saved figure to: %s\n', out_pdf);
