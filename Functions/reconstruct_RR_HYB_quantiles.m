function Phi_q = reconstruct_RR_HYB_quantiles(draws, q_vec)
% RECONSTRUCT_RR_HYB_QUANTILES  Posterior quantiles of the full TV
% coefficient Phi_t for the HYBRID bilinear sampler.
%
%   Phi_{i,1,t}    = Psi_bar_{i,1} + s_0_{i} * theta_0_{i,t}              (intercept)
%   Phi_{i,k>1,t}  = Psi_bar_{i,k} + S_l_m_{i,k-1} * (A_t B_t')_{i,k-1}    (lags)
%
% Inputs:
%   draws.Psi_bar : Mc x N x K           K = N*P + 1
%   draws.A       : Mc x T x N x R
%   draws.B       : Mc x T x K_lag x R   K_lag = K - 1
%   draws.S_l_m   : Mc x N x K_lag
%   draws.s_0     : Mc x N
%   draws.theta_0 : Mc x N x T
%   q_vec         : 1 x nQ quantile levels
%
% Output:
%   Phi_q : N x K x T x nQ

[Mc, N, K] = size(draws.Psi_bar);
T  = size(draws.A, 2);
R  = size(draws.A, 4);
K_lag = K - 1;
nQ = numel(q_vec);

assert(size(draws.B, 3) == K_lag, ...
       'reconstruct_RR_HYB_quantiles: expected B to be Mc x T x %d x R.', K_lag);

Phi_q = zeros(N, K, T, nQ);
for t = 1:T
    % Intercept column (k = 1)
    Phi_int = squeeze(draws.Psi_bar(:, :, 1)) ...
            + draws.s_0 .* squeeze(draws.theta_0(:, :, t));      % Mc x N
    Phi_q(:, 1, t, :) = permute(quantile(Phi_int, q_vec, 1), [2 3 1]);

    % Lag columns (k = 2..K)
    A_t = reshape(draws.A(:, t, :, :), Mc, N,     R);
    B_t = reshape(draws.B(:, t, :, :), Mc, K_lag, R);
    AB  = zeros(Mc, N, K_lag);
    for r = 1:R
        AB = AB + reshape(A_t(:,:,r), Mc, N, 1) .* reshape(B_t(:,:,r), Mc, 1, K_lag);
    end
    Phi_lag = draws.Psi_bar(:, :, 2:end) + draws.S_l_m .* AB;     % Mc x N x K_lag
    Phi_q(:, 2:end, t, :) = permute(quantile(Phi_lag, q_vec, 1), [2 3 4 1]);
end
end
