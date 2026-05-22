function IFs_Phi = compute_RR_HYB_Phi_IFs(draws)
% COMPUTE_RR_HYB_Phi_IFs  Inefficiency factors of the reconstructed full TVP
% coefficient for the rank-R HYBRID bilinear sampler (intercept TV separated).
%
%   Phi_{i,1,t}    = Psi_bar_{i,1} + s_0_{i} * theta_0_{i,t}                   (intercept)
%   Phi_{i,k>1,t}  = Psi_bar_{i,k} + S_l_m_{i,k-1} * sum_r A_{t,i,r} B_{t,k-1,r}
%
% Both the (A,B) factorization and the loadings (s_0, S_l_m) are
% sign-unidentified individually, but the full Phi is identified. Computing
% IFs on Phi is the only honest mixing diagnostic for the TV component.
%
% Inputs (HYB sampler outputs):
%   draws.Psi_bar : Mc x N x K           K = N*P + 1
%   draws.A       : Mc x T x N x R
%   draws.B       : Mc x T x K_lag x R   K_lag = K - 1
%   draws.S_l_m   : Mc x N x K_lag
%   draws.s_0     : Mc x N
%   draws.theta_0 : Mc x N x T
%
% Output:
%   IFs_Phi : N x K x T

[Mc, N, K] = size(draws.Psi_bar);
T          = size(draws.A, 2);
R          = size(draws.A, 4);
K_lag      = K - 1;

assert(size(draws.B, 3) == K_lag, ...
       'compute_RR_HYB_Phi_IFs: expected B to be Mc x T x %d x R.', K_lag);

IFs_Phi = zeros(N, K, T);
for t = 1:T
    % --- Intercept column (k = 1) ---
    Phi_int = squeeze(draws.Psi_bar(:, :, 1)) ...
            + draws.s_0 .* squeeze(draws.theta_0(:, :, t));     % Mc x N
    IFs_Phi(:, 1, t) = reshape(ineff_factor(Phi_int), N, 1);

    % --- Lag columns (k = 2..K) via the bilinear ---
    A_t = reshape(draws.A(:, t, :, :), Mc, N,     R);
    B_t = reshape(draws.B(:, t, :, :), Mc, K_lag, R);

    AB_t = zeros(Mc, N, K_lag);
    for r = 1:R
        AB_t = AB_t + reshape(A_t(:,:,r), Mc, N, 1) ...
                   .* reshape(B_t(:,:,r), Mc, 1, K_lag);
    end

    Phi_lag = draws.Psi_bar(:, :, 2:end) + draws.S_l_m .* AB_t;  % Mc x N x K_lag
    Phi_flat = reshape(Phi_lag, Mc, []);
    IFs_Phi(:, 2:end, t) = reshape(ineff_factor(Phi_flat), N, K_lag);
end
end
