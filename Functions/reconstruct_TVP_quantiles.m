function Phi_q = reconstruct_TVP_quantiles(draws, q_vec)
% RECONSTRUCT_TVP_QUANTILES  Quantiles of Phi_t = Psi_bar + reshape(s .* theta_t, N, K).
%
%   Inputs:
%     draws.Psi_bar : Mc x N x K  (static coefficient matrix per draw)
%     draws.s       : Mc x M      (loading, M = N*K)
%     draws.theta   : Mc x M x T  (TV states)
%     q_vec         : 1 x nQ quantile levels in (0, 1)
%
%   Output:
%     Phi_q : N x K x T x nQ
%
%   Processes one t at a time to keep memory bounded (Mc x N x K per t).

[Mc, N, K] = size(draws.Psi_bar);
T  = size(draws.theta, 3);
M  = N * K;
nQ = numel(q_vec);

Phi_q = zeros(N, K, T, nQ);
for t = 1:T
    theta_t = reshape(draws.theta(:, :, t), Mc, M);
    s_theta = reshape(draws.s .* theta_t, Mc, N, K);
    Phi_t   = draws.Psi_bar + s_theta;                  % Mc x N x K
    Phi_q(:, :, t, :) = permute(quantile(Phi_t, q_vec, 1), [2 3 4 1]);
end
end
