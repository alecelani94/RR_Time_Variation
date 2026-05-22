%% eval_fs.m -- Full-sample evaluation
%
% Loads Output/<dataset>.mat (produced by main_fs.m) and produces, for the
% unrestricted TVP-VAR benchmark and the rank-R RR sampler at a SELECTED R:
%
%   1) ONE 6-column RNE table comparing TVP and RR side-by-side per class.
%   2) N x (N+1) panel of estimated intercepts and lag-1 coefficients:
%        TVP benchmark -> shaded band + DASHED median
%        RR R=R_sel    -> solid median (thicker) + solid band edges
%   3) N x K panel of OVERLAID loading-posterior histograms:
%        TVP -> s        (semi-transparent blue)
%        RR  -> S_l_m    (semi-transparent red)

clear;
clc;
close all;

addpath(genpath(fullfile(pwd, 'Functions')));

%% ---- Selections -------------------------------------------------------

dataset    = 'small_4_gs5';   % 'small_3' | 'small_4' | 'small_5' | 'medium' | 'large'
P_sel      = 2;               % lag order of the saved run to load
R_sel      = 2;               % which RR rank to compare to TVP (1..R_max)
rr_variant = 'RWAR';          % '2RW' | 'RWAR' -- which saved run to load

load(fullfile(pwd, 'Output', sprintf('%s_P%d_%s.mat', dataset, P_sel, rr_variant)));

if R_sel < 1 || R_sel > R_max
    error('eval_fs:badR', 'R_sel = %d outside valid range [1, %d].', R_sel, R_max);
end

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

%% ---- One combined 6-column RNE table (TVP | RR R_sel) ----------------

row_names  = {'Static',  'Loading', 'Sigma', 'Reconstructed'};
tvp_fields = {'Psi_bar', 's',       'Sigma', 'Phi'};
rr_fields  = {'Psi_bar', 'S_l_m',   'Sigma', 'Phi'};

n_rows = numel(row_names);
tab    = nan(n_rows, 6);
for c = 1:n_rows
    v = 100 ./ IFs_TVP.(tvp_fields{c})(:);
    v = v(isfinite(v));
    if ~isempty(v), tab(c, 1:3) = quantile(v, [0.05, 0.50, 0.95]); end

    v = 100 ./ IFs_RR{R_sel}.(rr_fields{c})(:);
    v = v(isfinite(v));
    if ~isempty(v), tab(c, 4:6) = quantile(v, [0.05, 0.50, 0.95]); end
end

T_compare = array2table(tab, ...
    'VariableNames', {'TVP_Lo90', 'TVP_Med', 'TVP_Hi90', ...
                      'RR_Lo90',  'RR_Med',  'RR_Hi90'}, ...
    'RowNames',      row_names);
fprintf('\nRNE (%% = 100/IF) -- TVP vs RR R=%d:\n', R_sel);
disp(T_compare);

%% ---- Figure 1: N x (N+1) coefs -- TVP vs RR overlay -----------------

red_face  = [1.00 0.70 0.70];   % TVP band fill
red_line  = [0.80 0.20 0.20];   % TVP median (dashed)
blue_line = [0.10 0.30 0.80];   % RR lines (solid)

Phi_q_RR_sel = Phi_q_RR{R_sel};

figure('Name', sprintf('%s: TVP benchmark vs RR(R=%d) -- intercepts + lag 1', dataset, R_sel), ...
       'Position', [50 50 1500 850]);
tiledlayout(N, N+1, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:N
    for k = 1:(N+1)
        nexttile;
        med_T = squeeze(Phi_q_TVP(i, k, :, i_med));
        lo_T  = squeeze(Phi_q_TVP(i, k, :, i_lo68));
        hi_T  = squeeze(Phi_q_TVP(i, k, :, i_hi68));
        med_R = squeeze(Phi_q_RR_sel(i, k, :, i_med));
        lo_R  = squeeze(Phi_q_RR_sel(i, k, :, i_lo68));
        hi_R  = squeeze(Phi_q_RR_sel(i, k, :, i_hi68));

        hold on;
        fill([plot_dates; flipud(plot_dates)], [lo_T; flipud(hi_T)], ...
             red_face, 'EdgeColor', 'none', 'FaceAlpha', 0.45);
        h_T = plot(plot_dates, med_T, '--', 'Color', red_line, 'LineWidth', 1.3);
        h_R = plot(plot_dates, med_R, '-',  'Color', blue_line, 'LineWidth', 1.6);
        plot(plot_dates, lo_R,  '-', 'Color', blue_line, 'LineWidth', 0.8);
        plot(plot_dates, hi_R,  '-', 'Color', blue_line, 'LineWidth', 0.8);
        hold off;

        if k == 1
            ttl = sprintf('%s: intercept', Names{i});
        else
            ttl = sprintf('%s on %s(t-1)', Names{i}, Names{k-1});
        end
        title(ttl, 'Interpreter', 'none', 'FontSize', 9);
        xtickangle(45);
        try recessionplot; end %#ok<TRYNC>

        if i == 1 && k == 1
            legend([h_T, h_R], {'Benchmark', 'RR'}, ...
                   'Location', 'northeast', 'FontSize', 7);
        end
    end
end

%% ---- Figure 2: N x K loading posterior -- TVP vs RR overlaid ---------

M_load = size(draws_TVP.s, 2);
K_s    = M_load / N;
draws_RR_sel = draws_RR{R_sel};

tvp_fill = [0.80 0.30 0.30];   % red  (Benchmark, matches Figure 1)
rr_fill  = [0.20 0.40 0.80];   % blue (RR, matches Figure 1)

figure('Name', sprintf('%s: loading posterior -- TVP (blue) vs RR R=%d (red)', dataset, R_sel), ...
       'Position', [50 50 1500 850]);
tiledlayout(N, K_s, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:N
    for k = 1:K_s
        nexttile;
        col_idx = (k-1)*N + i;
        v_tvp   = draws_TVP.s(:, col_idx);
        v_rr    = squeeze(draws_RR_sel.S_l_m(:, i, k));

        vmin  = min([v_tvp; v_rr]);
        vmax  = max([v_tvp; v_rr]);
        edges = linspace(vmin, vmax, 41);

        hold on;
        h_tvp = histogram(v_tvp, 'BinEdges', edges, 'Normalization', 'pdf', ...
                  'FaceColor', tvp_fill, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        h_rr  = histogram(v_rr,  'BinEdges', edges, 'Normalization', 'pdf', ...
                  'FaceColor', rr_fill,  'EdgeColor', 'none', 'FaceAlpha', 0.5);
        xline(0, 'k--', 'LineWidth', 0.8);
        hold off;

        title(loading_title(Names, N, k, i), ...
              'Interpreter', 'none', 'FontSize', 8);
        set(gca, 'YTick', []);

        if i == 1 && k == 1
            legend([h_tvp, h_rr], {'Benchmark', 'RR'}, ...
                   'Location', 'northeast', 'FontSize', 7);
        end
    end
end


%% ====================== helpers ======================================

function ttl = loading_title(Names, N, k, i)
    if k == 1
        ttl = sprintf('%s: intercept', Names{i});
    else
        lag = ceil((k-1)/N);
        j   = mod(k-2, N) + 1;
        ttl = sprintf('%s on %s(t-%d)', Names{i}, Names{j}, lag);
    end
end
