function summary = diagnostics_TVP_rank_budget(draws_TVP, N, P)
% DIAGNOSTICS_TVP_RANK_BUDGET  Two diagnostics on unrestricted TVP draws to
% verify the rank-budget diagnosis for the hybrid spec.
%
% Reconstructs the TV coefficient matrix at each draw and each t,
%   Phi_tilde_t = reshape(s .* theta_t, N, K),    K = N*P + 1,
% and computes:
%
%   D1: rho_int(t) = ||Phi_tilde_t(:,1)||^2 / ||Phi_tilde_t||_F^2
%       (overall) and per equation
%        rho_int_row(i, t) = Phi_tilde_t(i,1)^2 / ||Phi_tilde_t(i, :)||^2.
%
%   D2: singular values of the lag block Phi_tilde_t(:, 2:end) at each t,
%       and the share of squared singular values captured by the top-R
%       (R = 1, 2, 3, 5).
%
% Output is summarized to posterior quantiles (no Mc-sized arrays returned)
% so the saved .mat stays small.
%
% Inputs:
%   draws_TVP : struct with fields
%       .s     : Mc x M
%       .theta : Mc x M x T
%   N, P : VAR dimensions
%
% Output (summary):
%   .q             : 1 x 5 quantile levels [0.10 0.25 0.50 0.75 0.90]
%   .rho_int_q     : T x 5 quantiles of D1 overall
%   .rho_int_row_q : N x T x 5 quantiles of D1 per equation
%   .sv_lag_q      : T x r x 5 quantiles of singular values (r = min(N, N*P))
%   .topR_share    : struct with fields R1, R2, R3, R5 each T x 5

[Mc, M, T] = size(draws_TVP.theta);
K = N * P + 1;
assert(M == N * K, 'M = N*K mismatch in draws_TVP.theta');

r = min(N, N*P);

q     = [0.10, 0.25, 0.50, 0.75, 0.90];
nq    = numel(q);

% Mc x T arrays for D1 (cheap: scalar per (d, t))
rho_int     = zeros(Mc, T);
rho_int_row = zeros(Mc, N, T);

% Mc x T x r for D2 (singular values per (d, t))
sv_lag = zeros(Mc, T, r);

fprintf('Computing diagnostics: Mc = %d, T = %d, N = %d, K = %d ...\n', ...
        Mc, T, N, K);

t0 = tic;
for d = 1:Mc
    s_d = squeeze(draws_TVP.s(d, :)).';                  % M x 1
    for t = 1:T
        theta_dt  = squeeze(draws_TVP.theta(d, :, t)).'; % M x 1
        Phi_tilde = reshape(s_d .* theta_dt, N, K);      % N x K (TV component)

        % --- D1 overall ---
        intsq = sum(Phi_tilde(:, 1).^2);
        totsq = sum(Phi_tilde(:).^2);
        rho_int(d, t) = intsq / max(totsq, eps);

        % --- D1 per equation ---
        rowsq_int = Phi_tilde(:, 1).^2;                   % N x 1
        rowsq_tot = sum(Phi_tilde.^2, 2);                 % N x 1
        rho_int_row(d, :, t) = (rowsq_int ./ max(rowsq_tot, eps)).';

        % --- D2: SVD of lag block ---
        sv_lag(d, t, :) = svd(Phi_tilde(:, 2:end));
    end

    if mod(d, max(1, floor(Mc/10))) == 0
        fprintf('  draw %d / %d  (elapsed %.1fs)\n', d, Mc, toc(t0));
    end
end

%% ---- posterior quantiles (collapse Mc dim) ----------------------------

summary.q = q;

% D1 overall: Mc x T -> T x nq
summary.rho_int_q = squeeze(quantile(rho_int, q, 1)).';     % T x nq

% D1 per row: Mc x N x T -> N x T x nq
summary.rho_int_row_q = permute(quantile(rho_int_row, q, 1), [2 3 1]);

% D2 singular values: Mc x T x r -> T x r x nq
summary.sv_lag_q = permute(quantile(sv_lag, q, 1), [2 3 1]);

% --- Top-R share of squared singular values ---
sv_lag_sq = sv_lag.^2;                                       % Mc x T x r
total_sq  = sum(sv_lag_sq, 3);                               % Mc x T
total_sq  = max(total_sq, eps);

summary.topR_share = struct();
for R_test = [1, 2, 3, 5]
    if R_test <= r
        top_sq    = sum(sv_lag_sq(:, :, 1:R_test), 3);       % Mc x T
        share     = top_sq ./ total_sq;                       % Mc x T
        share_q   = squeeze(quantile(share, q, 1)).';         % T x nq
        summary.topR_share.(sprintf('R%d', R_test)) = share_q;
    end
end

end
