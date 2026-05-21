function IFs_Phi = compute_RR_Phi_IFs(draws)
% COMPUTE_RR_PHI_IFS  Inefficiency factors of the reconstructed full TVP
% coefficient for the rank-R bilinear sampler:
%
%   Phi_{i,k,t} = Psi_bar_{ik} + S_l_m_{ik} * sum_r A_{t,i,r} * B_{t,k,r}.
%
% Both factorizations (A,B) and the loading S_l_m are unidentified, but the
% full TV coefficient Phi is. Computing IFs on Phi is the only honest mixing
% diagnostic for the TV component.
%
% Inputs:
%   draws.Psi_bar : Mc x N x K
%   draws.A       : Mc x T x N x R
%   draws.B       : Mc x T x K x R
%   draws.S_l_m   : Mc x N x K
%
% Output:
%   IFs_Phi : N x K x T
%
% Processes one t at a time so peak memory stays at Mc x N x K.

[Mc, N, K] = size(draws.Psi_bar);
T          = size(draws.A, 2);
R          = size(draws.A, 4);

IFs_Phi = zeros(N, K, T);
for t = 1:T
    % A_t, B_t at fixed t across all draws
    A_t = reshape(draws.A(:, t, :, :), Mc, N, R);
    B_t = reshape(draws.B(:, t, :, :), Mc, K, R);

    % AB_t(d, i, k) = sum_r A_t(d, i, r) * B_t(d, k, r)
    AB_t = zeros(Mc, N, K);
    for r = 1:R
        AB_t = AB_t + reshape(A_t(:,:,r), Mc, N, 1) ...
                   .* reshape(B_t(:,:,r), Mc, 1, K);
    end

    Phi_t   = draws.Psi_bar + draws.S_l_m .* AB_t;        % Mc x N x K
    Phi_flat = reshape(Phi_t, Mc, []);
    IFs_Phi(:, :, t) = reshape(ineff_factor(Phi_flat), N, K);
end
end
