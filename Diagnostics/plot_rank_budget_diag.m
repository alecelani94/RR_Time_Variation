function plot_rank_budget_diag(summary, Names, Ydates)
% PLOT_RANK_BUDGET_DIAG  Local-only plots for the rank-budget diagnostics.
%
% Designed to be run on a downloaded .mat output of main_diag.m on a local
% machine (NOT on a headless server). Produces four figures:
%
%   Fig 1: D1 overall -- intercept share of TV signal over time (median + IQR).
%   Fig 2: D1 per equation -- one line per row, posterior median over t.
%   Fig 3: D2 top-R share over time for R in {1,2,3,5}.
%   Fig 4: D2 singular value decay at three representative t (early, mid, late).
%
% Inputs:
%   summary : struct from diagnostics_TVP_rank_budget.m
%   Names   : N x 1 cell array of variable names
%   Ydates  : T x 1 datetime vector (post-burn dates)

% summary.q layout: e.g. [0.10 0.25 0.50 0.75 0.90]
q   = summary.q;
i50 = find(q == 0.50);
i25 = find(q == 0.25);
i75 = find(q == 0.75);
i10 = find(q == 0.10);
i90 = find(q == 0.90);

% Dates: drop the initial P observations from Ydates (the sampler does)
T_post = size(summary.rho_int_q, 1);
dates  = Ydates(end - T_post + 1 : end);

%% ---- Fig 1: D1 overall ------------------------------------------------

figure('Name', 'D1: intercept TV share (overall)', 'Color', 'w');
hold on;
fill([dates; flipud(dates)], ...
     [summary.rho_int_q(:, i10); flipud(summary.rho_int_q(:, i90))], ...
     [0.85 0.85 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
fill([dates; flipud(dates)], ...
     [summary.rho_int_q(:, i25); flipud(summary.rho_int_q(:, i75))], ...
     [0.70 0.70 0.90], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
plot(dates, summary.rho_int_q(:, i50), 'b-', 'LineWidth', 1.5);
yline(0.40, 'k--', '0.40 reference');
ylabel('rho_{int}(t) = ||\phi_{0,t}||^2 / ||\Theta_t||_F^2');
xlabel('t');
title('D1: intercept share of TV signal (posterior median + IQR + 80% band)');
grid on;
ylim([0 1]);
hold off;

%% ---- Fig 2: D1 per equation ------------------------------------------

figure('Name', 'D1: intercept TV share per equation', 'Color', 'w');
rho_row_med = squeeze(summary.rho_int_row_q(:, :, i50));   % N x T

colors = lines(size(rho_row_med, 1));
hold on;
for i = 1:size(rho_row_med, 1)
    plot(dates, rho_row_med(i, :), '-', 'Color', colors(i, :), ...
         'LineWidth', 1.5, 'DisplayName', Names{i});
end
yline(0.50, 'k--', 'HandleVisibility', 'off');
ylabel('rho^{int}_{i,t} per equation (posterior median)');
xlabel('t');
title('D1: intercept TV share by equation');
legend('Location', 'best');
grid on;
ylim([0 1]);
hold off;

%% ---- Fig 3: D2 top-R share over time ---------------------------------

figure('Name', 'D2: top-R share of squared singular values', 'Color', 'w');
hold on;
labels = {'R = 1', 'R = 2', 'R = 3', 'R = 5'};
fields = {'R1', 'R2', 'R3', 'R5'};
colors = lines(4);
for k = 1:numel(fields)
    if isfield(summary.topR_share, fields{k})
        plot(dates, summary.topR_share.(fields{k})(:, i50), ...
             '-', 'Color', colors(k, :), 'LineWidth', 1.5, ...
             'DisplayName', labels{k});
    end
end
yline(0.85, 'k--', '0.85 reference', 'HandleVisibility', 'off');
ylabel('Cumulative share of squared singular values (posterior median)');
xlabel('t');
title('D2: rank concentration of the lag block of \Theta_t');
legend('Location', 'best');
grid on;
ylim([0 1.05]);
hold off;

%% ---- Fig 4: D2 singular value decay at three representative t --------

figure('Name', 'D2: singular value decay (snapshots)', 'Color', 'w');
T = numel(dates);
ts = unique([1, round(T/2), T]);   % early, mid, late

hold on;
markers = {'o-', 's-', '^-'};
for k = 1:numel(ts)
    t = ts(k);
    sv_med = squeeze(summary.sv_lag_q(t, :, i50));    % r x 1
    plot(sv_med, markers{k}, 'LineWidth', 1.5, ...
         'DisplayName', sprintf('t = %d (%s)', t, datestr(dates(t), 'mmm yyyy')));
end
ylabel('Singular value (posterior median)');
xlabel('Index r');
title('D2: singular value decay of \Theta_t(:, lag) at three t');
legend('Location', 'best');
grid on;
hold off;

end
