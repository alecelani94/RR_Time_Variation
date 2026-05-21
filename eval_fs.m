clear;
clc;
close all;

addpath(genpath(fullfile(pwd, 'Functions')));

dataset = 'small_3';   % 'small_3' | 'small_4' | 'small_5' | 'medium' | 'large'

load(fullfile(pwd, 'Output', sprintf('%s.mat', dataset)));

%% ---- Sizes and quantile indices ---------------------------------------

N = size(Phi_q_TVP, 1);

idx_q  = @(p) find(abs(qoi - p) < 1e-10, 1);
i_med  = idx_q(0.50);
i_lo68 = idx_q(0.16);  i_hi68 = idx_q(0.84);

if any(cellfun(@isempty, {i_med, i_lo68, i_hi68}))
    error('eval_fs:missingQuantiles', ...
          'qoi must contain 0.16, 0.50, 0.84 for the 68%% band.');
end

plot_dates = Ydates(P+1:end);
plot_dates = plot_dates(:);

%% ---- Table: RNE summary per class -----------------------------------
% Inefficiency rescaled to RNE % = 100/IF so values are comparable across
% classes (and across runs of different chain length). Higher = better.

classes = {'Psi_bar', 'theta',   's',       'Sigma', 'Phi'};
pretty  = {'Static',  'Pure TV', 'St devs', 'Sigma', 'Reconstructed'};

n_cls   = numel(classes);
RNE_tab = nan(n_cls, 3);   % [5%, median, 95%]
for c = 1:n_cls
    name = classes{c};
    if isfield(IFs_TVP, name)
        v = 100 ./ IFs_TVP.(name)(:);
        v = v(isfinite(v));
        if ~isempty(v)
            RNE_tab(c, :) = quantile(v, [0.05, 0.50, 0.95]);
        end
    end
end

T_RNE = array2table(RNE_tab, ...
    'VariableNames', {'Lo90', 'Median', 'Hi90'}, ...
    'RowNames',      pretty);
fprintf('\nRelative numerical efficiency per class (RNE %% = 100 / IF):\n');
disp(T_RNE);

%% ---- Figure 2: intercept + lag-1 coefs with 68/90 bands --------------

red_band = [0.90 0.30 0.30];   % 68% band

figure('Name', sprintf('TVP coefs (%s): intercept and lag 1', dataset), ...
       'Position', [50 50 1500 850]);
tiledlayout(N, N+1, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:N
    for k = 1:(N+1)
        nexttile;
        med  = squeeze(Phi_q_TVP(i, k, :, i_med));
        lo68 = squeeze(Phi_q_TVP(i, k, :, i_lo68));
        hi68 = squeeze(Phi_q_TVP(i, k, :, i_hi68));

        hold on;
        fill([plot_dates; flipud(plot_dates)], [lo68; flipud(hi68)], red_band, ...
             'EdgeColor', 'none', 'FaceAlpha', 0.45);
        plot(plot_dates, med, 'r-', 'LineWidth', 1.3);
        hold off;

        if k == 1
            ttl = sprintf('%s: intercept', Names{i});
        else
            ttl = sprintf('%s on %s(t-1)', Names{i}, Names{k-1});
        end
        title(ttl, 'Interpreter', 'none', 'FontSize', 9);
        xtickangle(45);
        try recessionplot; end %#ok<TRYNC>
    end
end
sgtitle(sprintf('%s: estimated TVP coefficients - intercept and lag 1 (median, 68%% band)', dataset), ...
        'FontSize', 12, 'Interpreter', 'none');

%% ---- Figure 3: posterior histograms of loading s ---------------------
% s is stored as Mc x M with M = N*K, column-stack of an N x K matrix:
% col 1 = intercept loading, cols 2..K = lag loadings.

M_load = size(draws_TVP.s, 2);
K_s    = M_load / N;

figure('Name', sprintf('Posterior of loading s (%s)', dataset), ...
       'Position', [50 50 1500 850]);
tiledlayout(N, K_s, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:N
    for k = 1:K_s
        nexttile;
        col_idx = (k-1)*N + i;       % vec(N x K) column-major -> M
        v = draws_TVP.s(:, col_idx);
        histogram(v, 40, 'FaceColor', [0.20 0.40 0.80], 'EdgeColor', 'none');
        xline(0, 'k--', 'LineWidth', 0.8);

        if k == 1
            ttl = sprintf('%s: intercept', Names{i});
        else
            lag     = ceil((k-1) / N);
            var_idx = mod(k-2, N) + 1;
            ttl = sprintf('%s on %s(t-%d)', Names{i}, Names{var_idx}, lag);
        end
        title(ttl, 'Interpreter', 'none', 'FontSize', 8);
        set(gca, 'YTick', []);
    end
end
sgtitle(sprintf('%s: posterior of loading s (bimodality = FS-Wagner sign flip alive)', dataset), ...
        'FontSize', 12, 'Interpreter', 'none');
