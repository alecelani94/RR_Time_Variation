%% var_evolution_plot.m
% Plots the variance evolution of the per-element TV coefficient under
% three specifications, on a common horizon T = 250:
%
%   1. Unrestricted RW (target benchmark): Var(theta_{ij,t}) = t.
%   2. Symmetric RW x RW bilinear approx:  Var(tilde theta_{ij,t}) = t^2 / T.
%   3. AR(1) x RW bilinear refinement:     Var(tilde theta_{ij,t})
%                                          = t * (1 - rho^(2t)) / (1 - rho^(2T)),
%      under the MATCH-AT-T calibration
%      v^a = (1 - rho^2) / (1 - rho^(2T)) * (T/R)^(1/2),  v^b = (RT)^(-1/2).
%      This calibration makes Var = T exactly at the endpoint for every rho.
%
% Saves the figure as PDF + PNG inside the Overleaf folder for direct
% inclusion via \includegraphics.

clear; 
clc; 
close all;

T = 250;
t = 1:T;

% --- 1. Unrestricted RW target ---
var_target = t;

% --- 2. Symmetric RW x RW (current bilinear, calibrated to hit T at endpoint) ---
var_sym = t.^2 / T;

% --- 3. AR(1) x RW for several persistence values ---
rho_grid = [0.95, 0.99, 0.999];
var_ar   = zeros(numel(rho_grid), T);
for k = 1:numel(rho_grid)
    rho = rho_grid(k);
    var_ar(k, :) = t .* (1 - rho.^(2*t)) ./ (1 - rho.^(2*T));
end

%% ---- Plot ----------------------------------------------------------------

fig = figure('Position', [100, 100, 900, 560]);
hold on;

% Target: black solid, thick
plot(t, var_target, 'k-', 'LineWidth', 2.2, ...
     'DisplayName', 'Target RW');

% Symmetric: red dashed
plot(t, var_sym,    '--', 'Color', [0.80 0.20 0.20], 'LineWidth', 1.8, ...
     'DisplayName', 'RW \times RW');

% AR(1) variants: distinct colors
colors_ar = [0.10 0.55 0.20;   % green for 0.95
             0.10 0.30 0.80;   % blue  for 0.99
             0.55 0.20 0.70];  % purple for 0.999
for k = 1:numel(rho_grid)
    plot(t, var_ar(k, :), '-', 'Color', colors_ar(k, :), 'LineWidth', 1.6, ...
         'DisplayName', sprintf('AR(1) \\times RW,  \\rho = %.3f', rho_grid(k)));
end

hold off;
grid on;
xlim([0, T]);
ylim([0, T*1.05]);
xlabel('t', 'FontSize', 12);
legend('Location', 'northwest', 'FontSize', 12);
set(gca, 'FontSize', 11);

%% ---- Save into Overleaf folder ------------------------------------------

out_dir = fullfile(pwd, 'Overleaf');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

exportgraphics(fig, fullfile(out_dir, 'var_evolution.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(out_dir, 'var_evolution.png'), 'Resolution', 200);
fprintf('Saved: var_evolution.{pdf,png} in %s\n', out_dir);
