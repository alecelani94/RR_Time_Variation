function [V_phi0, C_Psi] = construct_Minnesota(scales, P, lambda_const, lambda_decay)
% CONSTRUCT_MINNESOTA  Minnesota prior constants for the static VAR
% coefficients of a TVP-VAR(P), with lambda_own / lambda_cross FACTORED
% OUT so they can be sampled hierarchically.
%
% Inputs
%   scales       : N x 1 -- AR(P0) residual variances (typical: P0 = 4)
%   P            : VAR lag order
%   lambda_const : scalar -- intercept prior variance multiplier (FIXED)
%   lambda_decay : scalar -- lag decay exponent (FIXED)
%
% Outputs
%   V_phi0       : N x 1 -- prior variance for the static intercept
%                  V_phi0(i) = lambda_const * scales(i)
%   C_Psi        : N x K with K = N*P -- constant scaling matrix
%                  C_Psi(i, (l-1)*N + j) = (scales(i)/scales(j)) / l^lambda_decay
%
% Under the hierarchical Minnesota, the full prior variance of
% Psi_bar(i, (l-1)*N + j) is then  lambda_kind * C_Psi(i, ...) where
% lambda_kind is lambda_own when i == j and lambda_cross otherwise.

N = numel(scales);
K = N * P;

V_phi0 = lambda_const * scales(:);

C_Psi = zeros(N, K);
for l = 1:P
    for j = 1:N
        col = (l-1)*N + j;
        for i = 1:N
            C_Psi(i, col) = (scales(i)/scales(j)) / (l^lambda_decay);
        end
    end
end
end
