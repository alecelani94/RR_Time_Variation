function Phi_q = reconstruct_RR_quantiles(draws, q_vec)
% RECONSTRUCT_RR_QUANTILES  Quantiles of Phi_t = Psi_bar + S_l_m .* (A_t * B_t').
%
%   Inputs:
%     draws.Psi_bar : Mc x N x K
%     draws.A       : Mc x T x N x R
%     draws.B       : Mc x T x K x R
%     draws.S_l_m   : Mc x N x K
%     q_vec         : 1 x nQ
%
%   Output:
%     Phi_q : N x K x T x nQ

[Mc, N, K] = size(draws.Psi_bar);
T  = size(draws.A, 2);
R  = size(draws.A, 4);
nQ = numel(q_vec);

Phi_q = zeros(N, K, T, nQ);
for t = 1:T
    A_t = reshape(draws.A(:, t, :, :), Mc, N, R);
    B_t = reshape(draws.B(:, t, :, :), Mc, K, R);
    AB  = zeros(Mc, N, K);
    for r = 1:R
        AB = AB + reshape(A_t(:,:,r), Mc, N, 1) .* reshape(B_t(:,:,r), Mc, 1, K);
    end
    Phi_t = draws.Psi_bar + draws.S_l_m .* AB;
    Phi_q(:, :, t, :) = permute(quantile(Phi_t, q_vec, 1), [2 3 4 1]);
end
end
