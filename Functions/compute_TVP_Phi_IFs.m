function IFs_Phi = compute_TVP_Phi_IFs(draws)
% COMPUTE_TVP_PHI_IFS  Inefficiency factors of the reconstructed full TVP
% coefficient Phi_{i,k,t} = Psi_bar_{ik} + s_{(k-1)*N+i} * theta_{(k-1)*N+i, t}.
%
% Sign-invariant (s and theta both flip together under FS-Wagner), so this
% is the diagnostic to read for true TV mixing -- unlike IF on s or theta
% alone, which the sign flip artificially drives to ~100% RNE.
%
% Inputs:
%   draws.Psi_bar : Mc x N x K
%   draws.s       : Mc x M       (M = N*K)
%   draws.theta   : Mc x M x T
%
% Output:
%   IFs_Phi : N x K x T  (one IF per reconstructed coefficient cell)
%
% Processes one t at a time to bound memory at Mc x N x K.

[Mc, N, K] = size(draws.Psi_bar);
T = size(draws.theta, 3);
M = N * K;

IFs_Phi = zeros(N, K, T);
for t = 1:T
    theta_t  = reshape(draws.theta(:, :, t), Mc, M);
    s_theta  = reshape(draws.s .* theta_t, Mc, N, K);
    Phi_t    = draws.Psi_bar + s_theta;            % Mc x N x K
    Phi_flat = reshape(Phi_t, Mc, []);             % Mc x (N*K)
    IFs_Phi(:, :, t) = reshape(ineff_factor(Phi_flat), N, K);
end
end
